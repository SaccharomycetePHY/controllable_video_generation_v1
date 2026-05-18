import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

  
"""
超参数设置
"""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLE_STEPS = 20  # 采样步数
VISUALIZE_RESULTS = True  # 是否保存可视化结果
ann_frame_idx = 0  # the frame index we interact with
"""
初始化模型
"""
# 初始化 SAM2 视频模型
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# 初始化 SAM2 图像模型
sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)

# mask_generator = SAM2AutomaticMaskGenerator(
#     model=sam2,
#     points_per_side=64,
#     points_per_batch=128,
#     pred_iou_thresh=0.7,
#     stability_score_thresh=0.92,
#     stability_score_offset=0.7,
#     box_nms_thresh=0.7,
#     crop_n_points_downscale_factor=2,
#     min_mask_region_area=25.0,
# )

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=25.0,
    use_m2m=True,
)

def process_folder(folder_path):
    # 检查是否已存在object masks文件
    object_masks_file = os.path.join(folder_path, "object_masks.npz")
    if os.path.exists(object_masks_file):
        print(f"已存在object masks文件: {object_masks_file}，跳过处理")
        return

    if VISUALIZE_RESULTS:
        visualized_mask_dir = os.path.join(folder_path, "object_mask")
        os.makedirs(visualized_mask_dir, exist_ok=True)

    # 加载人体掩码
    human_masks = np.load(os.path.join(folder_path, "human_masks.npz"))['masks'].squeeze(1)

    # 加载深度图
    depths = np.load(os.path.join(folder_path, "depths.npz"))['arr_0']

    video_dir = os.path.join(folder_path, "frames")

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)

    human_avg_depth = np.mean(depths[ann_frame_idx][np.where(human_masks[ann_frame_idx])])
    foreground_mask = (depths[ann_frame_idx] > human_avg_depth) & (1-human_masks[ann_frame_idx])

    image = Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))
    image = np.array(image.convert("RGB"))
    masks = mask_generator.generate(image)
    
    masks_foreground = []

    for mask in masks:
        # depth = depths[ann_frame_idx] * (1 - human_masks[ann_frame_idx]) 
        depth = depths[ann_frame_idx]
        mask_avg_depth = np.mean(depth[np.where(mask['segmentation'])])
        if mask_avg_depth > human_avg_depth:
            masks_foreground.append(mask)
    
    for object_id, mask in enumerate(masks_foreground):
        labels = np.array([1], dtype=np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            mask=mask['segmentation']
        )

    # 对所有帧进行处理并保存压缩的掩码
    all_masks = []
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        combined_mask = np.zeros_like(out_mask_logits[0].cpu().numpy(), dtype=np.uint8)
        human_mask = human_masks[out_frame_idx]
        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy().astype(np.uint8)
            # 去除与人体重叠的部分
            mask = mask & (1 - human_mask)
            combined_mask[mask > 0] = i + 1  # 使用不同的数字表示不同的物体
        all_masks.append(combined_mask)
        
        if VISUALIZE_RESULTS:
            # 可视化mask并保存为图片
            visualized_mask = combined_mask.squeeze()
            # 为每个物体分配不同的颜色
            color_mask = np.zeros((visualized_mask.shape[0], visualized_mask.shape[1], 3), dtype=np.uint8)
            unique_ids = np.unique(visualized_mask)
            for obj_id in unique_ids[1:]:  # 跳过背景（0）
                color = np.random.randint(0, 256, size=3)
                color_mask[visualized_mask == obj_id] = color
            cv2.imwrite(os.path.join(visualized_mask_dir, f"{out_frame_idx:05d}.jpg"), color_mask)
        
    # 将所有掩码保存为单个压缩文件
    np.savez_compressed(os.path.join(folder_path, "object_masks.npz"), masks=all_masks)
    
    print(f"已保存object masks到 {os.path.join(folder_path, 'object_masks.npz')}")
    
    # 检查是否成功生成掩码图片
    if VISUALIZE_RESULTS:
        mask_files = os.listdir(visualized_mask_dir)
        if len(mask_files) > 0:
            print(f"成功生成 {len(mask_files)} 个掩码图片")
        else:
            print("警告：未生成任何掩码图片")


def main():
    root_dir = "/amax/zyude/sample_video/BV1FT411A7yz"
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, 'clip_440-626')
        if os.path.isdir(folder_path):
            print(f"正在处理文件夹: {folder_path}")
            process_folder(folder_path)
            print(f"处理完成: {folder_path}")

if __name__ == "__main__":
    main()