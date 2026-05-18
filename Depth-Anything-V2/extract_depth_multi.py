import cv2
import torch
import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import time
from depth_anything_v2.dpt import DepthAnythingV2
import matplotlib
import torch.multiprocessing as mp
from tqdm import tqdm
cmap = matplotlib.colormaps.get_cmap('Spectral_r')
print(torch.cuda.is_available())
print('end')

# DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# DEVICE = 'cpu'

def init_model(device):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl'  # or 'vits', 'vitb', 'vitg'
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(device).eval()
    return model

def process_videos(gpu_id, videos, total_gpus):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    model = init_model(device)
    for video_dir, clip_name, clip_dir in tqdm(videos, desc=f"处理GPU {device}的视频"):
        try:
            video_path = os.path.join(clip_dir, "clip.mp4")
            cap = cv2.VideoCapture(video_path)
            out_path = os.path.join(clip_dir, "depth.npy")

            num = 0
            start = time.time()
            all_arry = []
            while cap.isOpened():
                num += 1
                ret, frame = cap.read()
                if not ret:
                    break

                depth = model.infer_image(frame)  # HxW raw depth map in numpy
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                all_arry.append(depth)
            all_arry = np.array(all_arry)
            np.save(out_path,all_arry)
            print(all_arry.shape[:])

            cap.release()
            end = time.time()
            print((end-start)/num)
        except Exception as e:
            print(e)
            continue

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = args.input_dir
    all_videos = []
    for video_name in os.listdir(input_dir):
        video_dir = os.path.join(input_dir, video_name)
        if os.path.isdir(video_dir):
            for clip_name in os.listdir(video_dir):
                clip_base_name = clip_name[:-4] if clip_name.endswith(".mp4") else clip_name
                clip_dir = os.path.join(video_dir, clip_base_name)
                if os.path.exists(os.path.join(clip_dir, "depth.npy")):
                    print(f"已处理完成: {clip_dir}")
                    continue
                if os.path.exists(os.path.join(clip_dir, "object.mp4")) or os.path.exists(os.path.join(clip_dir, "object_masks.npz")):
                    print(f"已完成object处理: {clip_dir}")
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
    main()
