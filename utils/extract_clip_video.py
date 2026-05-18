import os
import shutil
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True, type=str)
    parser.add_argument('--dst_dir', required=True, type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir

    # 遍历源目录下的每个视频文件夹
    for video_list in glob.glob(os.path.join(src_dir, '*')):
        for video_name in os.listdir(video_list):
            video_dir = os.path.join(video_list, video_name)
            print(video_name)
            if not os.path.isdir(video_dir):
                continue
                
            clips_dir = os.path.join(video_dir, 'clips')
            if not os.path.exists(clips_dir):
                continue

            # 遍历clips目录下的所有片段
            for clip_file in os.listdir(clips_dir):
                if clip_file.startswith('clip_') and clip_file.endswith('.mp4'):
                    # 获取片段id
                    clip_id = clip_file[5:-4]  # 去掉'clip_'前缀和'.mp4'后缀
                    
                    # 创建目标目录
                    dst_clip_dir = os.path.join(dst_dir, video_name, f'clip_{clip_id}')
                    
                    # 如果目标文件夹已存在则跳过
                    if os.path.exists(dst_clip_dir):
                        print(f"跳过已存在的目标文件夹: {dst_clip_dir}")
                        continue
                        
                    os.makedirs(dst_clip_dir, exist_ok=True)
                    
                    # 复制clip视频
                    src_clip = os.path.join(clips_dir, f'clip_{clip_id}.mp4')
                    dst_clip = os.path.join(dst_clip_dir, 'clip.mp4')
                    if os.path.exists(src_clip):
                        shutil.copy2(src_clip, dst_clip)
                        
                    # 复制emoca视频
                    src_emoca = os.path.join(clips_dir, f'emoca_{clip_id}.mp4')
                    dst_emoca = os.path.join(dst_clip_dir, 'emoca.mp4')
                    if os.path.exists(src_emoca):
                        shutil.copy2(src_emoca, dst_emoca)
                        
                    # 复制bbox文件
                    src_bbox = os.path.join(clips_dir, 'bbox', f'clip_{clip_id}_bbox.pkl')
                    dst_bbox = os.path.join(dst_clip_dir, 'bbox.pkl')
                    if os.path.exists(src_bbox):
                        shutil.copy2(src_bbox, dst_bbox)

    print("文件复制完成")

if __name__ == '__main__':
    main()
