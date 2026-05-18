import os
import cv2
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy

"""
超参数设置
"""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

"""
初始化模型
"""
# 初始化Grounding DINO模型
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

# 初始化SAM2模型
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
image_predictor = SAM2ImagePredictor(sam2_image_model)

def process_video(video_path, output_dir):
    # 创建输出目录
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # 创建人物和物体的输出目录
    human_dir = os.path.join(output_dir, "human")
    object_dir = os.path.join(output_dir, "object")
    mask_dir = os.path.join(output_dir, "masks")  # 新增mask保存目录
    os.makedirs(human_dir, exist_ok=True)
    os.makedirs(object_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)  # 创建mask保存目录
    
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
    
    # 加载深度信息
    depths_path = os.path.join(os.path.dirname(video_path), "depths.npz")
    depths = np.load(depths_path)['arr_0']
    
    # COCO数据集物品集
    coco_objects = "person.bicycle.menu.car.motorcycle.airplane.bus.train.truck.boat.traffic light.fire hydrant.stop sign.parking meter.bench.bird.cat.dog.horse.sheep.cow.elephant.bear.zebra.giraffe.backpack.umbrella.handbag.tie.suitcase.frisbee.skis.snowboard.sports ball.kite.baseball bat.baseball glove.skateboard.surfboard.tennis racket.bottle.wine glass.cup.fork.knife.spoon.bowl.banana.apple.sandwich.orange.broccoli.carrot.hot dog.pizza.donut.cake.chair.couch.potted plant.bed.dining table.toilet.tv.laptop.mouse.remote.keyboard.cell phone.microwave.oven.toaster.sink.refrigerator.book.clock.vase.scissors.teddy bear.hair drier.toothbrush."

    # 初始化视频预测器状态
    inference_state = video_predictor.init_state(video_path=frames_dir, offload_video_to_cpu=True, async_loading_frames=True)
    step = 20  # Grounding DINO预测器的采样步长
    
    sam2_masks = MaskDictionaryModel()
    PROMPT_TYPE_FOR_VIDEO = "mask"
    objects_count = 0
    
    for start_frame_idx in range(0, frame_count, step):
        print(f"处理第 {start_frame_idx} 帧")
        frame_path = os.path.join(frames_dir, f"{start_frame_idx:05d}.jpg")
        image_source, image = load_image(frame_path)
        
        # 使用Grounding DINO检测所有物体
        boxes, _, labels = predict(
            model=grounding_model,
            image=image,
            caption=coco_objects,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )
        
        # 使用SAM2获取掩码
        image_predictor.set_image(image_source)
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )

        # convert the mask shape to (n, H, W)
        if masks.ndim == 2:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        mask_dict = MaskDictionaryModel(promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"mask_{start_frame_idx:05d}.npy")
        mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(DEVICE), box_list=boxes, label_list=labels)
        
        objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
        
        video_predictor.reset_state(inference_state)
        
        if len(mask_dict.labels) == 0:
            print(f"第 {start_frame_idx} 帧未检测到物体，跳过")
            continue
        
        for object_id, object_info in mask_dict.labels.items():
            _, _, _ = video_predictor.add_new_mask(
                inference_state,
                start_frame_idx,
                object_id,
                object_info.mask,
            )
        
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
            frame_masks = MaskDictionaryModel()
            
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0)
                object_info = ObjectInfo(instance_id=out_obj_id, mask=out_mask[0], class_name=mask_dict.get_target_class_name(out_obj_id))
                object_info.update_box()
                frame_masks.labels[out_obj_id] = object_info
                frame_masks.mask_name = f"mask_{out_frame_idx:05d}.npy"
                frame_masks.mask_height = out_mask.shape[-2]
                frame_masks.mask_width = out_mask.shape[-1]
            
            video_segments[out_frame_idx] = frame_masks
            sam2_masks = copy.deepcopy(frame_masks)
        
        # 处理每一帧的掩码
        for frame_idx, frame_masks_info in video_segments.items():
            human_mask = np.zeros((frame_masks_info.mask_height, frame_masks_info.mask_width), dtype=bool)
            object_mask = np.zeros((frame_masks_info.mask_height, frame_masks_info.mask_width), dtype=bool)
            human_count = 0
            object_count = 0
            for obj_id, obj_info in frame_masks_info.labels.items():
                mask_bool = obj_info.mask.cpu().numpy().astype(bool)
                avg_depth = np.mean(depths[frame_idx][mask_bool])
                print(obj_info.class_name, avg_depth)
                if obj_info.class_name == "person":
                    human_mask |= mask_bool
                    human_count += 1
                elif not np.any(human_mask) or avg_depth > np.mean(depths[frame_idx][human_mask]):
                    object_mask |= mask_bool
                    object_count += 1
            
            # 可视化
            frame = cv2.imread(os.path.join(frames_dir, f"{frame_idx:05d}.jpg"))
            human_vis = frame.copy()
            human_vis[~human_mask] = [255, 255, 255]  # 白色背景
            human_vis[human_mask] = frame[human_mask]  # 保留原始图像中的人物部分
            cv2.imwrite(os.path.join(human_dir, f"human_{frame_idx:05d}.jpg"), human_vis)
            
            object_vis = frame.copy()
            object_vis[~object_mask] = [255, 255, 255]  # 白色背景
            object_vis[object_mask] = frame[object_mask]  # 保留原始图像中的物体部分
            cv2.imwrite(os.path.join(object_dir, f"object_{frame_idx:05d}.jpg"), object_vis)
            
            # 保存mask
            human_mask_vis = (human_mask * 255).astype(np.uint8)
            object_mask_vis = (object_mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(mask_dir, f"human_mask_{frame_idx:05d}.png"), human_mask_vis)
            cv2.imwrite(os.path.join(mask_dir, f"object_mask_{frame_idx:05d}.png"), object_mask_vis)
            
            print(f"第 {frame_idx} 帧包含的人数: {human_count}, 包含的物体数: {object_count}")
    
    print("视频处理完成")

def main():
    root_dir = "/amax/zyude/sample_video"
    for video_name in os.listdir(root_dir):
        video_dir = os.path.join(root_dir, video_name)
        if os.path.isdir(video_dir):
            for clip_name in os.listdir(video_dir):
                clip_dir = os.path.join(video_dir, clip_name)
                if os.path.isdir(clip_dir):
                    video_path = os.path.join(clip_dir, "clip.mp4")
                    if os.path.exists(video_path):
                        print(f"正在处理: {video_path}")
                        process_video(video_path, clip_dir)
                        print(f"处理完成: {video_path}")
                    else:
                        print(f"警告: 未找到视频文件 {video_path}")

if __name__ == "__main__":
    main()