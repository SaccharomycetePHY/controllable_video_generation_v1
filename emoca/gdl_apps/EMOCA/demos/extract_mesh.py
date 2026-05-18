import os
import shutil

video_dir = '/mnt/d/emoca/video_output2/EMOCA_v2_lr_mse_20'
# 过滤出文件夹
video_list = [item for item in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, item))]

for video_name in video_list:
    # 定义路径
    source_dir = f'{video_dir}/{video_name}/results'
    target_dir = f'/mnt/d/emoca/clips_t/{video_name}/mesh'

    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 遍历 source_dir 下的所有子文件夹
    for root, dirs, files in os.walk(source_dir):
        for dir_name in dirs:
            # 构建 obj 文件的完整路径
            obj_file = os.path.join(root, dir_name, 'mesh_coarse.obj')
            if os.path.exists(obj_file):
                # 新的文件名
                new_file_name = f"{dir_name}.obj"
                # 目标文件路径
                new_file_path = os.path.join(target_dir, new_file_name)
                # 移动并重命名文件
                shutil.copy(obj_file, new_file_path)
                print(f"Moved: {obj_file} -> {new_file_path}")
