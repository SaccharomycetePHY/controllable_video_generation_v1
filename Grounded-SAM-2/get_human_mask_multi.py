import os
import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
import argparse
import torch.multiprocessing as mp

"""
超参数设置
"""
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
TEXT_PROMPT = "human."
VISUALIZE_MASKS = False  # 新增参数，控制是否可视化mask

"""
初始化模型
"""
def init_models(device):
    # 初始化 Grounding DINO 模型
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=device
    )

    # 初始化 SAM2 模型
    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    return grounding_model, video_predictor, image_predictor

def process_video(video_path, output_dir, grounding_model, video_predictor, image_predictor, device):
    # 创建输出目录
    frames_dir = os.path.join(output_dir, "frames")
    
    os.makedirs(frames_dir, exist_ok=True)
    
    if VISUALIZE_MASKS:
        visualized_mask_dir = os.path.join(output_dir, "human_mask")
        os.makedirs(visualized_mask_dir, exist_ok=True)
    
    # 提取视频帧
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f"{frame_count:05d}.jpg"), frame)
        frame_count += 1
    cap.release()
    
    # 初始化视频预测器状态
    inference_state = video_predictor.init_state(video_path=frames_dir)
    
    # 对第一帧进行处理
    first_frame_path = os.path.join(frames_dir, "00000.jpg")
    image_source, image = load_image(first_frame_path)
    
    # 使用 Grounding DINO 获取边界框
    print(device)
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=device
    )
    
    # 处理边界框
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    # # 使用 SAM2 获取掩码
    # image_predictor.set_image(image_source)
    # masks, scores, logits = image_predictor.predict(
    #     point_coords=None,
    #     point_labels=None,
    #     box=input_boxes,
    #     multimask_output=False,
    # )
    # if masks.ndim == 4:
    #     masks = masks.squeeze(1)
    
    # sample the positive points from mask for each objects
    # all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

    # 修复：定义OBJECTS
    OBJECTS = labels

    # for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
    #     labels = np.ones((points.shape[0]), dtype=np.int32)
    #     _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
    #         inference_state=inference_state,
    #         frame_idx=0,  # 修复：使用0作为第一帧的索引
    #         obj_id=object_id,
    #         points=points,
    #         labels=labels,
    #     )

    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=object_id,
            box=box,
        )

    # 对所有帧进行处理并保存压缩的掩码
    all_masks = []
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        combined_mask = np.zeros_like(out_mask_logits[0].cpu().numpy(), dtype=np.uint8)
        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy().astype(np.uint8)
            combined_mask |= mask
        all_masks.append(combined_mask)
        
        if VISUALIZE_MASKS:
            # 可视化mask并保存为图片
            visualized_mask = combined_mask.squeeze()
            visualized_mask = (visualized_mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(visualized_mask_dir, f"{out_frame_idx:05d}.jpg"), visualized_mask)

    # 将所有掩码保存为单个压缩文件
    np.savez_compressed(os.path.join(output_dir, "human_masks.npz"), masks=all_masks)
    if os.path.exists(os.path.join(output_dir, "human_masks.npz")):
        print(f"成功生成掩码文件: {os.path.join(output_dir, 'human_masks.npz')}")
    else:
        print(f"警告：未生成掩码文件: {os.path.join(output_dir, 'human_masks.npz')}")
    # 检查是否成功生成掩码图片
    if VISUALIZE_MASKS:
        mask_files = os.listdir(visualized_mask_dir)
        if len(mask_files) > 0:
            print(f"成功生成 {len(mask_files)} 个掩码图片")
        else:
            print("警告：未生成任何掩码图片")

def process_videos(gpu_id, videos, total_gpus):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    
    grounding_model, video_predictor, image_predictor = init_models(device)

    for video_dir, clip_name, clip_dir in tqdm(videos, desc=f"处理GPU {device}的视频"):
        try:
            # 检查是否存在处理到一半的视频
            if os.path.isdir(clip_dir) and not os.path.exists(os.path.join(clip_dir, "human_masks.npz")):
                print(f"发现处理到一半的视频: {clip_dir}")
                video_path = os.path.join(clip_dir, "clip.mp4")
                if os.path.exists(video_path):
                    process_video(video_path, clip_dir, grounding_model, video_predictor, image_predictor, device)
                    print(f"重新处理完成: {video_path}")
                else:
                    print(f"警告: 未找到视频文件 {video_path}")
            elif clip_name.endswith(".mp4"):
                # 处理新的视频文件
                os.makedirs(clip_dir, exist_ok=True)
                
                # 移动并重命名视频文件
                old_video_path = os.path.join(video_dir, clip_name)
                new_video_path = os.path.join(clip_dir, "clip.mp4")
                os.rename(old_video_path, new_video_path)
                
                # 处理视频
                print(f"正在处理: {new_video_path}")
                process_video(new_video_path, clip_dir, grounding_model, video_predictor, image_predictor, device)
                print(f"处理完成: {new_video_path}")
        except Exception as e:
            print(f"处理视频 {clip_name} 时发生错误: {str(e)}")
            continue  # 继续处理下一个视频

    # 清理CUDA缓存
    torch.cuda.empty_cache()

def main(args):
    root_dir = args.root_dir
    all_videos = []
    for video_name in os.listdir(root_dir):
        video_dir = os.path.join(root_dir, video_name)
        if os.path.isdir(video_dir):
            for clip_name in os.listdir(video_dir):
                clip_base_name = clip_name[:-4] if clip_name.endswith(".mp4") else clip_name
                clip_dir = os.path.join(video_dir, clip_base_name)
                if os.path.exists(os.path.join(clip_dir, "human_masks.npz")):
                    print(f"已处理完成: {clip_dir}")
                    continue
                all_videos.append((video_dir, clip_name, clip_dir))

    print(f"总共处理的视频数量: {len(all_videos)}")
    # 多进程处理
    num_gpus = torch.cuda.device_count()
    videos_per_gpu = len(all_videos) // num_gpus
    processes = []
    
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * videos_per_gpu
        end_idx = start_idx + videos_per_gpu if gpu_id < num_gpus - 1 else len(all_videos)
        gpu_videos = all_videos[start_idx:end_idx]
        
        p = mp.Process(target=process_videos, args=(gpu_id, gpu_videos, num_gpus))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="多卡并行处理视频数据")
    parser.add_argument('--root_dir', required=True, type=str)
    args = parser.parse_args()
    main(args)