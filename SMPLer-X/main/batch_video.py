import os
import subprocess

root_folder = '/amax/zyude/results_10_6/'


for dirpath, _, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.endswith('.mp4'):
        
            video_path = os.path.join(dirpath, filename)
            
        
            command = f'CUDA_VISIBLE_DEVICES=7 python inference_video.py --video_path "{video_path}" --multi_person'
            
          
            print(f"正在处理: {video_path}")

            subprocess.run(command, shell=True)

print("所有视频处理完成。")
