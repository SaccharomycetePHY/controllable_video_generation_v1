import cv2
import os
import logging
import argparse
import pickle

from datetime import datetime

import numpy as np
from omegaconf import OmegaConf
from typing import Dict, Tuple

import torch
import decord

from decord import VideoReader
from torchvision.transforms import transforms

from transformers import AutoModel

from diffusers import AutoencoderKL, DDIMScheduler, AutoencoderKLTemporalDecoder
from diffusers.utils import check_min_version

from src.data.dwpose_utils.draw_pose import draw_pose
from src.models.rd_unet_normal import RealisDanceUnet
from src.pipelines.pipeline_normal import RealisDancePipeline
from src.utils.util import save_videos_grid


decord.bridge.set_bridge('torch')


def augmentation(frame, transform, state=None):
    if state is not None:
        torch.set_rng_state(state)
    return transform(frame)


# def simple_reader(ref_image_path, dwpose_path, hamer_path, smpl_path, sample_size, clip_size, max_length):
# def simple_reader(ref_image_path, hamer_path, smpl_path, bg_path, fg_path, sample_size, clip_size, max_length):
def simple_reader(raw_video_path, ref_image_path, hamer_path, smpl_path, normal_path, bg_path, fg_path, sample_size, clip_size, max_length):
    scale = (1.0, 1.0)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        # ratio is w/h
        transforms.RandomResizedCrop(
            sample_size, scale=scale,
            ratio=(sample_size[1] / sample_size[0], sample_size[1] / sample_size[0]), antialias=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    img_transform_ = transforms.Compose([
        # transforms.ToTensor(),
        # ratio is w/h
        transforms.RandomResizedCrop(
            sample_size, scale=scale,
            ratio=(sample_size[1] / sample_size[0], sample_size[1] / sample_size[0]), antialias=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    clip_transform = transforms.Compose([
        transforms.ToTensor(),
        # ratio is w/h
        transforms.RandomResizedCrop(
            clip_size, scale=scale,
            ratio=(clip_size[1] / clip_size[0], clip_size[1] / clip_size[0]), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406],  # used for dino
                             [0.229, 0.224, 0.225],  # used for dino
                             inplace=True),
    ])
    pose_transform = transforms.Compose([
        # ratio is w/h
        transforms.RandomResizedCrop(
            sample_size, scale=scale,
            ratio=(sample_size[1] / sample_size[0], sample_size[1] / sample_size[0]), antialias=True),
    ])
    normal_transform = transforms.Compose([
        # ratio is w/h
        transforms.RandomResizedCrop(
            sample_size, scale=(0.75, 1.0),  # 随机缩放范围在0.8-1.2之间
            ratio=(0.5 * sample_size[1]/sample_size[0], 2 * sample_size[1]/sample_size[0]),  # 允许长宽比在原始比例的0.75-1.25倍之间变化
            antialias=True),
    ])
    raw_video_reader = VideoReader(raw_video_path)
    if os.path.exists(hamer_path):
        hamer_reader = VideoReader(hamer_path)
    else:
        hamer_reader = None
    if os.path.exists(smpl_path):
        smpl_reader = VideoReader(smpl_path)
    else:
        smpl_reader = None
    normal_reader = VideoReader(normal_path)
    bg_reader = VideoReader(bg_path)
    fg_reader = VideoReader(fg_path)
    # with open(dwpose_path, 'rb') as pose_file:
    #     pose_list = pickle.load(pose_file)
    # assert len(hamer_reader) == len(smpl_reader) == len(pose_list)
    video_length = len(normal_reader)
    batch_index = range(0, video_length, 4)[:max_length]
    normal = normal_reader.get_batch(batch_index).permute(0, 3, 1, 2).contiguous() / 255.0

    if hamer_reader is not None:
        hamer = hamer_reader.get_batch(batch_index).permute(0, 3, 1, 2).contiguous() / 255.0
    else:
        hamer = torch.zeros_like(normal)
    try:
        smpl = smpl_reader.get_batch(batch_index).permute(0, 3, 1, 2).contiguous() / 255.0
    except:
        smpl = torch.zeros_like(normal)  # 创建一个与hamer相同大小的全0张量,代表全黑视频
    
    bg = bg_reader.get_batch(batch_index).permute(0, 3, 1, 2).contiguous() / 255.0
    fg = fg_reader.get_batch(batch_index).permute(0, 3, 1, 2).contiguous() / 255.0
    raw_video = raw_video_reader.get_batch(batch_index).permute(0, 3, 1, 2).contiguous() / 255.0

    # pose = [draw_pose(pose_list[batch_index[idx]], hamer.shape[-2], hamer.shape[-1], draw_face=False)
    #         for idx in range(len(batch_index))]
    # pose = torch.from_numpy(
    #     np.stack(pose, axis=0)).permute(0, 3, 1, 2).contiguous() / 255.0

    _ref_img = cv2.cvtColor(cv2.imread(ref_image_path), cv2.COLOR_BGR2RGB)
    state = torch.get_rng_state()
    ref_image_clip = augmentation(_ref_img, clip_transform, state)
    
    ######### 创建一个480x640的白色背景
    white_bg = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # 获取原始图片尺寸
    h, w = _ref_img.shape[:2]  # 修正获取图片尺寸的方式
    
    # 计算缩放比例
    scale = min(480/h, 640/w)
    if scale < 1:  # 只有当图片超出范围时才缩放
        new_h = int(h * scale)
        new_w = int(w * scale)
        _ref_img = cv2.resize(_ref_img, (new_w, new_h))
    else:
        new_h, new_w = h, w
        
    # 计算居中位置
    y_offset = (480 - new_h) // 2
    x_offset = (640 - new_w) // 2
    
    # 将图片贴到白色背景中间
    white_bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = _ref_img
    _ref_img = white_bg
    ######### 将图片贴到白色背景上

    ref_image = augmentation(_ref_img, img_transform, state)
    
    # pose = augmentation(pose, pose_transform, state)
    hamer = augmentation(hamer, pose_transform, state)
    smpl = augmentation(smpl, pose_transform, state)
    # normal = augmentation(normal, pose_transform, state)
    normal = augmentation(normal, normal_transform, state)
    bg = augmentation(bg, img_transform_, state)
    fg = augmentation(fg, img_transform_, state)
    raw_video = augmentation(raw_video, img_transform_, state)

    del hamer_reader
    del smpl_reader
    del normal_reader
    del bg_reader
    del fg_reader
    del raw_video_reader
    return (
        ref_image.unsqueeze(0),
        ref_image_clip.unsqueeze(0),
        # pose.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
        hamer.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
        smpl.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
        normal.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
        bg.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
        fg.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
        raw_video.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
    )


def main(
    output_dir: str,
    pretrained_model_path: str,
    pretrained_clip_path: str,
    ref_image_path: str,
    hamer_path: str,
    # dwpose_path: str,
    smpl_path: str,
    normal_path: str,
    bg_path: str,
    fg_path: str,
    raw_video_path: str,
    sample_size: Tuple,
    clip_size: Tuple,
    max_length: int,

    save_path: str,

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

    mixed_precision: str = "fp16",

    global_seed: int or str = 42,
    is_debug: bool = False,
    *args,
    **kwargs,
):
    ref_name = os.path.splitext(os.path.basename(ref_image_path))[0]
    # dwpose_name = os.path.splitext(os.path.basename(dwpose_path))[0]
    hamer_name = os.path.splitext(os.path.basename(hamer_path))[0]
    smpl_name = os.path.splitext(os.path.basename(smpl_path))[0]
    normal_name = os.path.splitext(os.path.basename(normal_path))[0]
    # output_name = f"r_{ref_name}_d_{dwpose_name}_h_{hamer_name}_s_{smpl_name}"
    output_name = f"r_{ref_name}_h_{hamer_name}_s_{smpl_name}"

    # check version
    check_min_version("0.30.0.dev0")

    if global_seed == "random":
        global_seed = int(datetime.now().timestamp()) % 65535

    seed = global_seed
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
    os.makedirs(os.path.join(
        output_dir, 'vis', 'mp4'), exist_ok=True)
    os.makedirs(os.path.join(
        output_dir, 'vis', 'gif'), exist_ok=True)
    os.makedirs(os.path.join(
        output_dir, 'samples', 'mp4'), exist_ok=True)
    os.makedirs(os.path.join(
        output_dir, 'samples', 'gif'), exist_ok=True)

    # Load scheduler, tokenizer and models
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
        image_finetune=False,
        unet_additional_kwargs=unet_additional_kwargs,
        pose_guider_kwargs=pose_guider_kwargs,
        clip_projector_kwargs=clip_projector_kwargs,
        fix_ref_t=fix_ref_t,
        fusion_blocks=fusion_blocks,
    )

    # Load pretrained unet weights
    logging.info(f"from checkpoint: {unet_checkpoint_path}")
    unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
    if "global_step" in unet_checkpoint_path:
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
    validation_pipeline.image_finetune = False
    validation_kwargs_container = {} if validation_kwargs is None else OmegaConf.to_container(validation_kwargs)
    if vae_slicing and 'SVD' not in pretrained_vae_path:
        validation_pipeline.enable_vae_slicing()

    # move to cuda
    vae.to("cuda")
    image_encoder.to("cuda")
    unet.to("cuda")
    validation_pipeline = validation_pipeline.to("cuda")

    # val_ref_image, val_ref_image_clip, val_pose, val_hamer, val_smpl = simple_reader(
    #     ref_image_path=ref_image_path,
    #     dwpose_path=dwpose_path,
    #     hamer_path=hamer_path,
    #     smpl_path=smpl_path,
    #     sample_size=sample_size,
    #     clip_size=clip_size,
    #     max_length=max_length,
    # )

    val_ref_image, val_ref_image_clip, val_hamer, val_smpl, val_normal, val_bg_image, val_fg_image, val_raw_video = simple_reader(
        raw_video_path=raw_video_path,
        ref_image_path=ref_image_path,
        hamer_path=hamer_path,
        smpl_path=smpl_path,
        normal_path=normal_path,
        bg_path=bg_path,
        fg_path=fg_path,
        sample_size=sample_size,
        clip_size=clip_size,
        max_length=max_length,
    )

    logging.info("***** Running validation *****")

    generator = torch.Generator(device=unet.device)
    generator.manual_seed(global_seed)

    height, width = sample_size

    val_ref_image = val_ref_image.to("cuda")
    val_ref_image_clip = val_ref_image_clip.to("cuda")
    # val_pose = val_pose.to("cuda")
    val_hamer = val_hamer.to("cuda")
    val_smpl = val_smpl.to("cuda")
    val_normal = val_normal.to("cuda")
    val_bg_image = val_bg_image.to("cuda")
    val_fg_image = val_fg_image.to("cuda")

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

    video_length = sample.shape[2]
    val_ref_image = val_ref_image.unsqueeze(2).repeat(1, 1, video_length, 1, 1)
    print(val_ref_image.shape, val_raw_video.shape, sample.shape)
    save_obj = torch.cat([
        (val_ref_image.cpu() / 2 + 0.5).clamp(0, 1),
        # (val_bg_image.cpu() / 2 + 0.5).clamp(0, 1),
        # (val_fg_image.cpu() / 2 + 0.5).clamp(0, 1),
        # val_hamer.cpu(),
        # val_smpl.cpu(),
        # val_normal.cpu(),
        sample.cpu(),
        (val_raw_video.cpu() / 2 + 0.5).clamp(0, 1),
    ], dim=-1)

    # save_path = f"{output_dir}/vis/mp4/{output_name}.mp4"
    # save_videos_grid(save_obj, save_path, fps=fps)
    # save_path = f"{output_dir}/vis/gif/{output_name}.gif"
    # save_videos_grid(save_obj, save_path, fps=fps)
    # save_videos_grid(save_obj, save_path, fps=fps)
    # sample_save_path = f"{output_dir}/samples/mp4/{output_name}.mp4"
    # save_videos_grid(sample.cpu(), sample_save_path, fps=fps)
    # sample_save_path = f"{output_dir}/samples/gif/{output_name}.gif"
    # save_videos_grid(sample.cpu(), sample_save_path, fps=fps, save_frame=save_frame)

    save_videos_grid(save_obj, save_path, fps=fps)

# outputs/stage2_normal-2024-11-28T14/checkpoints/checkpoint-iter-70000.ckpt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage2_syn.yaml")
    parser.add_argument("--output", type=str, default="infer_output")
    parser.add_argument("--ckpt", type=str, default="outputs/stage2_syn-2024-12-25T20/checkpoints/checkpoint-iter-100000.ckpt")
    parser.add_argument("--H", type=int, default=480)
    parser.add_argument("--W", type=int, default=640)
    parser.add_argument("--cH", type=int, default=240)
    parser.add_argument("--cW", type=int, default=320)
    parser.add_argument("--max-L", type=int, default=80)
    args = parser.parse_args()

    base_dir = "/zouyude/data/infer_data/infer_same_person"
    
    # 遍历所有子目录
    for subdir in os.listdir(base_dir):
        base_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(base_path):
            continue
            
        exp_config = OmegaConf.load(args.config)
        exp_config["output_dir"] = os.path.join(args.output, "infer_sameperson_syn_2")
        exp_config["unet_checkpoint_path"] = args.ckpt
        exp_config["raw_video_path"] = os.path.join(base_path, "clip.mp4")
        exp_config["save_path"] = os.path.join(exp_config["output_dir"], f"{subdir}.mp4")
        exp_config["ref_image_path"] = os.path.join(base_path, f"ref.png")
        exp_config["smpl_path"] = os.path.join(base_path, "smpl.mp4")
        exp_config["normal_path"] = os.path.join(base_path, "normal.mp4") 
        exp_config["hamer_path"] = os.path.join(base_path, "emoca.mp4")
        exp_config["bg_path"] = os.path.join(base_path, "inpaint_out.mp4")
        exp_config["fg_path"] = os.path.join(base_path, "object.mp4")
        exp_config["sample_size"] = (args.H, args.W)
        exp_config["clip_size"] = (args.cH, args.cW)
        exp_config["max_length"] = args.max_L
        exp_config["save_path"] = os.path.join(exp_config["output_dir"], f"{subdir}.mp4")
        exp_config["ref_image_path"] = os.path.join(base_path, f"ref.png")
        
        # 遍历当前目录下所有图片作为ref image
        ref_images = []
        for file in os.listdir(base_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # if file != "ref.png":  # 排除原始的ref.png
                ref_path = os.path.join(base_path, file)
                exp_config["save_path"] = os.path.join(exp_config["output_dir"], f"{subdir}_{file.split('.')[0]}.mp4")
                exp_config["ref_image_path"] = ref_path
                
                if os.path.exists(exp_config["save_path"]):
                    print(f"已处理完成: {exp_config['save_path']}")
                    continue
                
                main(**exp_config)
