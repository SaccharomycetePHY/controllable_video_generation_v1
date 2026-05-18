import os
import logging
import argparse
import subprocess

from datetime import datetime
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from typing import Dict

import torch
import torchvision
import torch.distributed as dist

from transformers import AutoModel

from diffusers import AutoencoderKL, DDIMScheduler, AutoencoderKLTemporalDecoder
from diffusers.utils import check_min_version

from src.models.rd_unet_normal import RealisDanceUnet
from src.pipelines.pipeline_normal import RealisDancePipeline
from src.utils.util import get_distributed_dataloader, save_videos_grid, sanity_check

from skimage import metrics
import lpips

from src.utils.fvd_metric import compute_fvd

import numpy as np
from scipy import linalg

def calculate_psnr(img1, img2):
    # Convert torch tensors to numpy arrays
    img1 = img1.cpu().numpy().transpose(1,2,0)
    img2 = img2.cpu().numpy().transpose(1,2,0)
    psnr = metrics.peak_signal_noise_ratio(img1, img2)
    # print(f"PSNR: {psnr}")
    return psnr

def calculate_ssim(img1, img2):
    # Convert torch tensors to numpy arrays 
    img1 = img1.cpu().numpy().transpose(1,2,0)
    img2 = img2.cpu().numpy().transpose(1,2,0)
    ssim = metrics.structural_similarity(img1, img2, channel_axis=-1, data_range=1.0)
    # print(f"SSIM: {ssim}")
    return ssim

def calculate_lpips(img1, img2, lpips_model):
    lpips_value = lpips_model(img1, img2)
    return lpips_value.item()

def calculate_video_metrics(gt, pred, lpips_model):

    fvd = 0
    fidvid = 0
    # fidvid = compute_fvd(gt, pred, 32, 'cuda', batch_size=4)

    psnr_list, ssim_list, lpips_list = [], [], []
    for i in range(gt.shape[2]):
        gt_i = gt[:,:,i,:,:].squeeze(0)  # 添加squeeze(0)来移除第一个维度
        pred_i = pred[:,:,i,:,:].squeeze(0)  # 添加squeeze(0)来移除第一个维度
        psnr = calculate_psnr(gt_i, pred_i)
        ssim = calculate_ssim(gt_i, pred_i)
        lpips = calculate_lpips(gt_i, pred_i, lpips_model)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpips)

    return fvd, fidvid, np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)

def calculate_img_metrics(gt, pred, lpips_model):
    gt = gt.squeeze(0)
    pred = pred.squeeze(0)
    psnr = calculate_psnr(gt, pred)
    ssim = calculate_ssim(gt, pred)
    lpips = calculate_lpips(gt, pred, lpips_model)
    return psnr, ssim, lpips

def init_dist(launcher="slurm", backend="nccl", port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == "pytorch":
        rank = int(os.environ["RANK"])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)

    elif launcher == "slurm":
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f"scontrol show hostname {node_list} | head -n1")
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["RANK"] = str(proc_id)
        port = os.environ.get("PORT", port)
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend=backend)
        print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; "
              f"node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")

    else:
        raise NotImplementedError(f"Not implemented launcher type: `{launcher}`!")

    return local_rank


def main(
    image_finetune: bool,
    launcher: str,

    output_dir: str,
    pretrained_model_path: str,
    pretrained_clip_path: str,

    validation_data: Dict,
    unet_checkpoint_path: str,
    validation_kwargs: Dict = None,
    fps: int = 8,
    save_frame: bool = False,
    train_cfg: bool = True,

    pretrained_vae_path: str = "",
    unet_additional_kwargs: Dict = None,
    noise_scheduler_kwargs: Dict = None,
    pose_guider_kwargs: Dict = None,
    fusion_blocks: str = "full",
    clip_projector_kwargs: Dict = None,
    fix_ref_t: bool = False,
    zero_snr: bool = False,
    v_pred: bool = False,
    vae_slicing: bool = False,
    
    num_workers: int = 4,
    validation_batch_size: int = 1,

    mixed_precision: str = "fp16",

    global_seed: int or str = 42,
    is_debug: bool = False,
    sanity_check_during_validation: bool = False,

    *args,
    **kwargs,
):
    # check version
    check_min_version("0.30.0.dev0")

    psnr_list = []
    ssim_list = []
    lpips_list = []
    fvd_list = []
    fidvid_list = []
    import lpips
    lpips_model = lpips.LPIPS(net="vgg")
    
    # Initialize distributed evaluation
    local_rank = init_dist(launcher=launcher)
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0

    if global_seed == "random":
        global_seed = int(datetime.now().timestamp()) % 65535

    seed = global_seed + global_rank
    torch.manual_seed(seed)

    # Logging folder
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Handle the output folder creation
    if is_main_process:
        if image_finetune:
            os.makedirs(os.path.join(
                output_dir, 'vis'), exist_ok=True)
            os.makedirs(os.path.join(
                output_dir, 'samples'), exist_ok=True)
        else:
            os.makedirs(os.path.join(
                output_dir, 'vis', 'mp4'), exist_ok=True)
            os.makedirs(os.path.join(
                output_dir, 'vis', 'gif'), exist_ok=True)
            os.makedirs(os.path.join(
                output_dir, 'samples', 'mp4'), exist_ok=True)
            os.makedirs(os.path.join(
                output_dir, 'samples', 'gif'), exist_ok=True)

    # Load scheduler, tokenizer and models
    if is_main_process:
        logging.info("Load scheduler, tokenizer and models.")
    if pretrained_vae_path != "":
        if 'SVD' in pretrained_vae_path:
            vae = AutoencoderKLTemporalDecoder.from_pretrained(pretrained_vae_path, subfolder="vae")
        else:
            vae = AutoencoderKL.from_pretrained(pretrained_vae_path, subfolder="sd-vae-ft-mse")
    else:
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")

    image_encoder = AutoModel.from_pretrained(pretrained_clip_path)

    noise_scheduler_kwargs_dict = OmegaConf.to_container(
        noise_scheduler_kwargs
    ) if noise_scheduler_kwargs is not None else {}
    if zero_snr:
        if is_main_process:
            logging.info("Enable Zero-SNR")
        noise_scheduler_kwargs_dict["rescale_betas_zero_snr"] = True
        if v_pred:
            noise_scheduler_kwargs_dict["prediction_type"] = "v_prediction"
            noise_scheduler_kwargs_dict["timestep_spacing"] = "linspace"
    noise_scheduler = DDIMScheduler.from_pretrained(
        pretrained_model_path,
        subfolder="scheduler",
        **noise_scheduler_kwargs_dict,
    )

    unet = RealisDanceUnet(
        pretrained_model_path=pretrained_model_path,
        image_finetune=image_finetune,
        unet_additional_kwargs=unet_additional_kwargs,
        pose_guider_kwargs=pose_guider_kwargs,
        clip_projector_kwargs=clip_projector_kwargs,
        fix_ref_t=fix_ref_t,
        fusion_blocks=fusion_blocks,
    )

    # Load pretrained unet weights
    if is_main_process:
        logging.info(f"from checkpoint: {unet_checkpoint_path}")
    unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
    if "global_step" in unet_checkpoint_path:
        if is_main_process:
            logging.info(f"global_step: {unet_checkpoint_path['global_step']}")
    state_dict = unet_checkpoint_path["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[7:]
        else:
            new_k = k
        new_state_dict[new_k] = state_dict[k]
    m, u = unet.load_state_dict(new_state_dict, strict=False)
    if is_main_process:
        logging.info(f"Load from checkpoint with missing keys:\n{m}")
        logging.info(f"Load from checkpoint with unexpected keys:\n{u}")

    # Freeze vae and image_encoder
    vae.eval()
    vae.requires_grad_(False)
    image_encoder.eval()
    image_encoder.requires_grad_(False)
    unet.eval()
    unet.requires_grad_(False)

    # Set validation pipeline
    validation_pipeline = RealisDancePipeline(
        unet=unet, vae=vae, image_encoder=image_encoder, scheduler=noise_scheduler)
    validation_pipeline.image_finetune = image_finetune
    validation_kwargs_container = {} if validation_kwargs is None else OmegaConf.to_container(validation_kwargs)
    if vae_slicing and 'SVD' not in pretrained_vae_path:
        validation_pipeline.enable_vae_slicing()

    # move to cuda
    vae.to(local_rank)
    image_encoder.to(local_rank)
    unet.to(local_rank)
    validation_pipeline = validation_pipeline.to(local_rank)

    # Get the validation dataloader
    validation_dataloader = get_distributed_dataloader(
        dataset_config=validation_data,
        batch_size=validation_batch_size,
        num_processes=num_processes,
        num_workers=num_workers,
        shuffle=False,
        global_rank=global_rank,
        seed=global_seed,)

    if is_main_process:
        logging.info("***** Running validation *****")
        logging.info(f"  Instantaneous validation batch size per device = {validation_batch_size}")

    generator = torch.Generator(device=unet.device)
    generator.manual_seed(global_seed)
    for val_batch in tqdm(validation_dataloader):
        # check sanity during validation
        if sanity_check_during_validation:
            if is_main_process:
                os.makedirs(f"{output_dir}/sanity_check/", exist_ok=True)
            sanity_check(val_batch, f"{output_dir}/sanity_check", image_finetune, global_rank)

        height, width = val_batch["hamer"].shape[-2:]
        if "image" in val_batch and isinstance(val_batch["image"], torch.Tensor):
            val_gt = val_batch["image"].to(local_rank)
        # val_pose = val_batch["pose"].to(local_rank)
        val_hamer = val_batch["hamer"].to(local_rank)
        val_smpl = val_batch["smpl"].to(local_rank)
        val_normal = val_batch["normal"].to(local_rank)
        # val_hamer = torch.zeros_like(val_hamer)
        # val_smpl = torch.zeros_like(val_smpl)
        val_ref_image = val_batch["ref_image"].to(local_rank)
        val_ref_image_clip = val_batch["ref_image_clip"].to(local_rank)

        val_bg_image = val_batch["bg_image"].to(local_rank)
        val_fg_image = val_batch["fg_image"].to(local_rank)

        # Predict the noise residual and compute loss
        # Mixed-precision training
        if mixed_precision in ("fp16", "bf16"):
            weight_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
        else:
            weight_dtype = torch.float32
        with torch.cuda.amp.autocast(
            enabled=mixed_precision in ("fp16", "bf16"),
            dtype=weight_dtype
        ):
            sample = validation_pipeline(
                # pose=val_pose,
                hamer=val_hamer,
                smpl=val_smpl,
                normal=val_normal,
                bg_image=val_bg_image,
                fg_image=val_fg_image,
                ref_image=val_ref_image,
                ref_image_clip=val_ref_image_clip,
                height=height, width=width,
                fake_uncond=not train_cfg,
                **validation_kwargs_container).videos
        # TODO: support more images per prompt
        num_images_per_prompt = 1
        for idx, data_id in enumerate(val_batch["data_key"]):
            samples = sample[idx*num_images_per_prompt:(idx+1)*num_images_per_prompt]
            # val_poses = val_pose[idx*num_images_per_prompt:(idx+1)*num_images_per_prompt]
            val_hamers = val_hamer[idx*num_images_per_prompt:(idx+1)*num_images_per_prompt]
            val_smpls = val_smpl[idx*num_images_per_prompt:(idx+1)*num_images_per_prompt]
            val_normals = val_normal[idx*num_images_per_prompt:(idx+1)*num_images_per_prompt]
            ref_images = val_ref_image[idx*num_images_per_prompt:(idx+1)*num_images_per_prompt]
            if not image_finetune:
                video_length = samples.shape[2]
                ref_images = ref_images.unsqueeze(2).repeat(1, 1, video_length, 1, 1)
            if "image" in val_batch and isinstance(val_batch["image"], torch.Tensor):
                val_gts = val_gt[idx*num_images_per_prompt:(idx+1)*num_images_per_prompt]
                save_obj = torch.cat([
                    (ref_images.cpu() / 2 + 0.5).clamp(0, 1),
                    (val_bg_image.cpu() / 2 + 0.5).clamp(0, 1),
                    (val_fg_image.cpu() / 2 + 0.5).clamp(0, 1),
                    # val_poses.cpu(),
                    val_hamers.cpu(),
                    val_smpls.cpu(),
                    val_normals.cpu(),
                    samples.cpu(),
                    (val_gts.cpu() / 2 + 0.5).clamp(0, 1),
                ], dim=-1)
            else:
                save_obj = torch.cat([
                    (ref_images.cpu() / 2 + 0.5).clamp(0, 1),
                    # val_poses.cpu(),
                    val_hamers.cpu(),
                    val_smpls.cpu(),
                    val_normals.cpu(),
                    samples.cpu(),
                ], dim=-1)
            data_id = data_id.replace('/', '_')
            if image_finetune:
                save_path = f"{output_dir}/vis/{data_id}_{global_rank}.png"
                torchvision.utils.save_image(save_obj, save_path, nrow=4)
                sample_save_path = f"{output_dir}/samples/{data_id}_{global_rank}_samples.png"
                torchvision.utils.save_image(samples.cpu(), sample_save_path, nrow=4)
            else:
                save_path = f"{output_dir}/vis/mp4/{data_id}_{global_rank}" + ".mp4"
                save_videos_grid(save_obj, save_path, fps=fps)
                save_path = f"{output_dir}/vis/gif/{data_id}_{global_rank}" + ".gif"
                save_videos_grid(save_obj, save_path, fps=fps)
                sample_save_path = f"{output_dir}/samples/mp4/{data_id}_{global_rank}_samples" + ".mp4"
                save_videos_grid(samples.cpu(), sample_save_path, fps=fps)
                sample_save_path = f"{output_dir}/samples/gif/{data_id}_{global_rank}_samples" + ".gif"
                save_videos_grid(samples.cpu(), sample_save_path, fps=fps, save_frame=save_frame)
            if image_finetune:
                psnr, ssim, lpips = calculate_img_metrics((val_gts.cpu() / 2 + 0.5).clamp(0, 1), samples.cpu(), lpips_model)
                print(f"{data_id}: PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}")
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                lpips_list.append(lpips)
            else:
                fvd, fidvid, psnr, ssim, lpips = calculate_video_metrics((val_gts.cpu() / 2 + 0.5).clamp(0, 1), samples.cpu(), lpips_model)
                print(f"{data_id}: FVD: {fvd}, fidvid: {fidvid}, PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}")
                fvd_list.append(fvd)
                fidvid_list.append(fidvid)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                lpips_list.append(lpips)
    print(f"FVD: {np.mean(fvd_list)}, fidvid: {np.mean(fidvid_list)}, PSNR: {np.mean(psnr_list)}, SSIM: {np.mean(ssim_list)}, LPIPS: {np.mean(lpips_list)}")
    with open(f"{output_dir}/metrics.txt", "a") as f:
        f.write(f"FVD: {np.mean(fvd_list)}, fidvid: {np.mean(fidvid_list)}, PSNR: {np.mean(psnr_list)}, SSIM: {np.mean(ssim_list)}, LPIPS: {np.mean(lpips_list)}\n")
    dist.destroy_process_group()


# outputs/stage1_hamer-test-2024-11-08T22/checkpoints/checkpoint-iter-46000.ckpt
# outputs/stage1_hamer-test-2024-11-08T22/checkpoints/checkpoint-iter-36000.ckpt
# outputs/stage2_normal-2024-11-11T23/checkpoints/checkpoint-iter-34000.ckpt
# outputs/stage1_hamer-test-2024-11-08T22/checkpoints/checkpoint-iter-44000.ckpt
# stage1_normal-2024-11-23T01 stage1_normal-2024-11-22T12

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage2_normal.yaml")
    parser.add_argument("--output", type=str, default="eval_outputs")
    parser.add_argument("--ckpt", type=str, default="outputs/stage2_normal-2024-11-28T14/checkpoints/checkpoint-iter-100000.ckpt")
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--sanity-check-during-validation", action="store_true")
    args = parser.parse_args()

    exp_config = OmegaConf.load(args.config)
    exp_config["output_dir"] = args.output
    exp_config["unet_checkpoint_path"] = args.ckpt

    exp_config["global_seed"] = 41
    exp_config["output_dir"] = os.path.join(args.output, "stage2_normal_aug/100000/eval",str(exp_config["global_seed"]))
    main(launcher=args.launcher, sanity_check_during_validation=args.sanity_check_during_validation,**exp_config)

    # for i in [84000, 94000, 104000, 114000, 120000, 124000]:
    #     exp_config["unet_checkpoint_path"] = f"outputs/stage1_normal-2024-11-22T12/checkpoints/checkpoint-iter-{i}.ckpt"
    #     exp_config["output_dir"] = os.path.join(args.output, f"stage1_normal_aug/{i}/eval",str(exp_config["global_seed"]))
    #     main(launcher=args.launcher, sanity_check_during_validation=args.sanity_check_during_validation,
    #         **exp_config)
