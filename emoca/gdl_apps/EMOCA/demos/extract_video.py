import os
import shutil
from pathlib import Path
import argparse
import subprocess

def get_video_duration(file_path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)

def count_frames(frames_dir):
    return len([f for f in os.listdir(frames_dir) if f.endswith('.jpg') or f.endswith('.png')])

def process_clips(mode='count'):
    root_dir = '/amax/zyude/results_10_6'
    target_dir = '/amax/zyude/video_10_6'
    total_clips = 0
    total_duration = 0
    total_original_duration = 0
    
    # 遍历根目录下的所有文件夹
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            # 遍历每个文件夹下的视频目录
            for video_dir in os.listdir(folder_path):
                video_path = os.path.join(folder_path, video_dir)
                clips_path = os.path.join(video_path, 'clips')
                
                if os.path.isdir(clips_path):
                    # 遍历clips目录中的所有视频文件
                    for clip in os.listdir(clips_path):
                        if clip.endswith('.mp4'):
                            total_clips += 1
                            source_path = os.path.join(clips_path, clip)
                            
                            if mode == 'copy':
                                # 创建目标目录
                                dest_dir = os.path.join(target_dir, video_dir)
                                Path(dest_dir).mkdir(parents=True, exist_ok=True)
                                
                                # 复制文件
                                dest_path = os.path.join(dest_dir, clip)
                                if not os.path.exists(dest_path):
                                    shutil.copy(source_path, dest_path)
                                    print(f'已复制: {source_path} -> {dest_path}')
                                else:
                                    print(f'目标文件已存在，跳过复制: {dest_path}')
                            
                            # 无论是否复制，都统计时长
                            duration = get_video_duration(source_path)
                            total_duration += duration
                            print(f'视频片段: {source_path}, 时长: {duration:.2f}秒')
                
                # 统计原始视频时长
                original_video_path = f'/amax/data/human_videos/bilibili/{folder}/{video_dir}.mp4'
                if os.path.exists(original_video_path):
                    original_duration = get_video_duration(original_video_path)
                    total_original_duration += original_duration
                    print(f'原始视频: {original_video_path}, 时长: {original_duration:.2f}秒')
                else:
                    print(f'原始视频不存在: {original_video_path}')

    print(f"总共处理 {total_clips} 个视频片段。")
    print(f"剪辑后总时长: {total_duration:.2f}秒 ({total_duration/3600:.2f}小时)")
    print(f"原始视频总时长: {total_original_duration:.2f}秒 ({total_original_duration/3600:.2f}小时)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理视频片段')
    parser.add_argument('--mode', choices=['count', 'copy', 'duration'], default='copy', help='选择模式：count（只计数）、copy（复制文件并统计时长）或duration（仅统计总时长）')
    args = parser.parse_args()

    process_clips(args.mode)
    print(f"视频片段处理完成。模式：{args.mode}")
