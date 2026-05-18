import os
import numpy as np
import cv2
import subprocess
import argparse
from tqdm import tqdm
import torch.multiprocessing as mp

def process_single_video(clip_base_path, frames_path, mask_path, output_video, video_type):
    # 读取遮罩文件
    masks = np.load(mask_path)['masks'].squeeze()
    
    # 读取所有应用的帧图
    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.png') or f.endswith('.jpg')])

    # 获取帧的宽和高
    example_frame = cv2.imread(os.path.join(frames_path, frame_files[0]))
    if example_frame is None:
        print(f"无法读取帧文件: {frame_files[0]}")
        return
    height, width, _ = example_frame.shape

    # 创建临时存储处理后帧的文件夹
    temp_frames_path = os.path.join(clip_base_path, f'temp_{video_type}_frames')
    os.makedirs(temp_frames_path, exist_ok=True)

    # 进行帧图的处理
    for idx, frame_file in enumerate(frame_files):
        # 读取当前帧
        frame = cv2.imread(os.path.join(frames_path, frame_file))
        if frame is None:
            print(f"无法读取帧文件: {frame_file}")
            continue

        # 创建白色背景
        white_background = np.ones_like(frame) * 255

        # 应用遮罩
        if idx < masks.shape[0]:
            mask = masks[idx].astype(bool)
            if mask.shape[:2] != frame.shape[:2]:
                print(f"{video_type}遮罩与帧尺寸不匹配: {frame_file}")
                print(masks.shape, frame.shape)
                continue
            processed_frame = np.where(mask[..., None], frame, white_background)
        else:
            processed_frame = white_background

        # 将处理后的帧保存为临时图像文件
        frame_path = os.path.join(temp_frames_path, f'{idx:04d}.png')
        cv2.imwrite(frame_path, processed_frame)
    
    # 使用 ffmpeg 合成视频
    subprocess.call(['ffmpeg', '-y', '-framerate', '30', '-i', os.path.join(temp_frames_path, '%04d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_video])

    # 删除临时帧文件夹
    for f in os.listdir(temp_frames_path):
        os.remove(os.path.join(temp_frames_path, f))
    os.rmdir(temp_frames_path)

def process_videos_batch(videos_batch, video_type):
    for clip_base_path, frames_path, mask_path, output_video in tqdm(videos_batch, desc=f"处理{video_type}视频"):
        process_single_video(clip_base_path, frames_path, mask_path, output_video, video_type)

def process_video(base_path, video_type):
    # 先收集所有需要处理的视频
    videos_to_process = []
    video_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    for video_name in video_dirs:
        clip_dirs = [d for d in os.listdir(os.path.join(base_path, video_name)) if os.path.isdir(os.path.join(base_path, video_name, d))]
        for clip_name in clip_dirs:
            clip_base_path = os.path.join(base_path, video_name, clip_name)
            frames_path = os.path.join(clip_base_path, 'frames')
            mask_path = os.path.join(clip_base_path, f'{video_type}_masks.npz')
            output_video = os.path.join(clip_base_path, f'{video_type}.mp4')

            if os.path.exists(output_video) or not os.path.exists(mask_path):
                print(f"跳过已处理的视频: {video_name} {clip_name}")
                continue
            
            if not os.path.exists(frames_path):
                # 创建frames目录
                os.makedirs(frames_path)
                video_path = os.path.join(clip_base_path, "clip.mp4")
                
                # 使用OpenCV读取视频并保存帧
                cap = cv2.VideoCapture(video_path)
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_path = os.path.join(frames_path, f"{frame_idx:05d}.png")
                    cv2.imwrite(frame_path, frame)
                    frame_idx += 1
                cap.release()
                print(f"已将视频分割为 {frame_idx} 帧，保存在 {frames_path}")
                
            videos_to_process.append((clip_base_path, frames_path, mask_path, output_video))

    print(f"共有{len(videos_to_process)}个视频需要处理")

    # 多进程并行处理
    num_processes = min(mp.cpu_count(), len(videos_to_process))
    batch_size = len(videos_to_process) // num_processes
    processes = []

    for i in range(num_processes):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size if i < num_processes - 1 else len(videos_to_process)
        batch = videos_to_process[start_idx:end_idx]
        
        p = mp.Process(target=process_videos_batch, args=(batch, video_type))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"视频生成完成：{video_type}.mp4")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='处理视频')
    parser.add_argument('--base_path', type=str, default='/zouyude/data/results_11_14', help='基础路径')
    parser.add_argument('--type', type=str, choices=['human', 'object', 'both'], default='both', help='处理类型 human、object 或 both')
    args = parser.parse_args()

    base_path = args.base_path
    
    if args.type == 'both':
        try:
            process_video(base_path, 'human')
            process_video(base_path, 'object')
        except Exception as e:
            print(f"处理视频时出错: {e}")
    else:
        process_video(base_path, args.type)