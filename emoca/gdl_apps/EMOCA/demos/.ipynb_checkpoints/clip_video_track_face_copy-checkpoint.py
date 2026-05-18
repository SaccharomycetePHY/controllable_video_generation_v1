import os
import sys
sys.path.append('/zouyude/controllable_video_generation/emoca')
import subprocess
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import pandas as pd
import numpy as np
import gc
from gdl_apps.EMOCA.utils.load import load_model
from gdl_apps.EMOCA.utils.io import test
from gdl.models.DecaFLAME import FLAME, FLAMETex, FLAME_mediapipe
from omegaconf import OmegaConf
import torch

flame_cfg = OmegaConf.load("/zouyude/controllable_video_generation/emoca/assets/EMOCA/models/EMOCA_v2_lr_mse_20/cfg.yaml")
flame_cfg = flame_cfg.coarse.model
flame = FLAME(flame_cfg)

def read_vertices_from_code(shape_code, exp_code, pose_code):
    shape_code = torch.tensor(shape_code, dtype=torch.float32).reshape(1, -1)
    exp_code = torch.tensor(exp_code, dtype=torch.float32).reshape(1, -1)
    pose_code = torch.tensor(pose_code, dtype=torch.float32).reshape(1, -1)
    
    verts, landmarks2d, landmarks3d = flame(shape_params=shape_code, expression_params=exp_code,
                                                          pose_params=pose_code)
    verts = verts.squeeze(0).cpu().numpy()
    vertices = []
    for x, y, z in verts:
        vertices.append((float(x), float(y), float(z)))
    return vertices
    # print(vertices)

def read_vertices_from_obj(obj_file_path):
    vertices = []
    try:
        with open(obj_file_path, 'r') as obj_file:
            for line in obj_file:
                if line.startswith('v '):
                    parts = line.strip().split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append((x, y, z))
        if vertices:
            return vertices
        else:
            return None
    except FileNotFoundError:
        return None

def calculate_avg_distance(vertices1, vertices2):
    if vertices1 and vertices2:
        if len(vertices1) != len(vertices2):
            print("警告：两个网格的顶点数量不一致。")
            min_length = min(len(vertices1), len(vertices2))
        else:
            min_length = len(vertices1)
        total_error = 0.0
        for i in range(min_length):
            v1 = vertices1[i]
            v2 = vertices2[i]
            diff_x = abs(v1[0] - v2[0])
            diff_y = abs(v1[1] - v2[1])
            diff_z = abs(v1[2] - v2[2])
            vertex_error = (diff_x + diff_y + diff_z) / 3
            total_error += vertex_error
        average_error = total_error / min_length
        return average_error
    return None

def calculate_max_distance(vertices1, vertices2):
    if vertices1 and vertices2:
        if len(vertices1) != len(vertices2):
            print("警告：两个网格的顶点数量不一致。")
            min_length = min(len(vertices1), len(vertices2))
        else:
            min_length = len(vertices1)
        max_error = 0.0
        for i in range(min_length):
            v1 = vertices1[i]
            v2 = vertices2[i]
            diff_x = abs(v1[0] - v2[0])
            diff_y = abs(v1[1] - v2[1])
            diff_z = abs(v1[2] - v2[2])
            vertex_error = (diff_x + diff_y + diff_z) / 3
            if vertex_error > max_error:
                max_error = vertex_error
        return max_error
    return None

def detect_shot_changes(results_dir, sizes, centers, threshold=0.02):
    frame_dirs = sorted([d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))])

    frame_pair = []
    clips = []
    current_clip_start = None
    last_vertices_dict = {}
    last_faces_count = 0
    last_frame_number = None

    current_frame_vertices_list = []
    current_frame_number = None

    sorted_sizes = sizes.copy()
    sorted_centers = centers.copy()

    for frame_dir in tqdm(frame_dirs):
        frame_number, face_id = frame_dir.split('_')
        face_id = int(face_id)
        
        if sizes[int(frame_number)-1][face_id] <= 80:
            continue

        # current_vertices = read_vertices_from_obj(os.path.join(results_dir, frame_dir, 'mesh_coarse.obj'))

        if os.path.exists(os.path.join(results_dir, frame_dir, 'mesh_coarse.obj')):
            current_vertices = read_vertices_from_obj(os.path.join(results_dir, frame_dir, 'mesh_coarse.obj'))
        else:
            shape_file = os.path.join(results_dir, frame_dir, 'shape.npy')
            exp_file = os.path.join(results_dir, frame_dir, 'exp.npy')
            pose_file = os.path.join(results_dir, frame_dir, 'pose.npy')

            shape_code = np.load(shape_file)
            exp_code = np.load(exp_file)
            pose_code = np.load(pose_file)

            current_vertices = read_vertices_from_code(shape_code, exp_code, pose_code)

        if current_vertices:
            if current_frame_number is None:
                current_frame_number = frame_number

            if frame_number == current_frame_number:
                current_frame_vertices_list.append(current_vertices)
            else:
                if last_frame_number and int(current_frame_number) != int(last_frame_number) + 1:
                    if current_clip_start is not None:
                        clips.append((current_clip_start, last_frame_number))
                    current_clip_start = current_frame_number
                    last_vertices_dict[current_frame_number] = current_frame_vertices_list[:]
                    last_faces_count = len(current_frame_vertices_list)
                    last_frame_number = current_frame_number
                    current_frame_vertices_list = [current_vertices]
                    current_frame_number = frame_number
                    continue

                if last_faces_count != len(current_frame_vertices_list) or int(current_frame_number) - int(last_frame_number) > 300:
                    if current_clip_start is not None:
                        clips.append((current_clip_start, last_frame_number))
                    current_clip_start = current_frame_number
                    last_vertices_dict[current_frame_number] = current_frame_vertices_list[:]
                    last_faces_count = len(current_frame_vertices_list)
                    last_frame_number = current_frame_number
                    current_frame_vertices_list = [current_vertices]
                    current_frame_number = frame_number
                    continue

                if last_faces_count > 0 and len(current_frame_vertices_list) > 0:
                    min_faces_count = min(last_faces_count, len(current_frame_vertices_list))
                    max_faces_count = max(last_faces_count, len(current_frame_vertices_list))
                    
                    distance_matrix = np.zeros((max_faces_count, max_faces_count))
                    for i in range(max_faces_count):
                        for j in range(max_faces_count):
                            if i < len(last_vertices_dict[last_frame_number]) and j < len(current_frame_vertices_list):
                                distance_matrix[i][j] = calculate_max_distance(last_vertices_dict[last_frame_number][i], current_frame_vertices_list[j])
                            else:
                                distance_matrix[i][j] = float('inf')
                    
                    # 使用匈牙利算法进行最优匹配
                    from scipy.optimize import linear_sum_assignment
                    row_ind, col_ind = linear_sum_assignment(distance_matrix)
                    
                    # 根据匹配结果重新排序当前帧的人脸
                    current_frame_vertices_list = [current_frame_vertices_list[i] for i in col_ind if i < len(current_frame_vertices_list)]
                    
                    # 重新排序sizes和centers
                    current_sizes = sorted_sizes[int(current_frame_number)-1]
                    current_centers = sorted_centers[int(current_frame_number)-1]
                    sorted_current_sizes = [current_sizes[i] for i in col_ind if i < len(current_sizes)]
                    sorted_current_centers = [current_centers[i] for i in col_ind if i < len(current_centers)]
                    
                    # 更新排序后的sizes和centers
                    sorted_sizes[int(current_frame_number)-1] = sorted_current_sizes
                    sorted_centers[int(current_frame_number)-1] = sorted_current_centers
                    
                    for i in range(min_faces_count):
                        min_distance = distance_matrix[row_ind[i]][col_ind[i]]
                        min_size_change = abs(sorted_sizes[int(last_frame_number)-1][row_ind[i]] - sorted_current_sizes[i])
                        frame_pair.append((last_frame_number, current_frame_number, min_distance, min_size_change))
                        
                        if min_distance > threshold or min_size_change > 30:
                            if current_clip_start is not None:
                                clips.append((current_clip_start, last_frame_number))
                            current_clip_start = current_frame_number
                            break

                last_vertices_dict[current_frame_number] = current_frame_vertices_list[:]
                last_faces_count = len(current_frame_vertices_list)
                last_frame_number = current_frame_number

                current_frame_vertices_list = [current_vertices]
                current_frame_number = frame_number
        else:
            print("current_vertices is None")

    if current_clip_start is not None and last_frame_number and int(current_clip_start) <= int(last_frame_number):
        clips.append((current_clip_start, last_frame_number))

    return clips, frame_pair, sorted_sizes, sorted_centers

def save_clips_to_file(clips, output_file):
    if not os.path.exists(Path(output_file).parent):
        os.makedirs(Path(output_file).parent)

    with open(output_file, 'w') as file:
        for clip in clips:
            file.write(f"{clip[0]} {clip[1]}\n")
    print(f"剪辑注释已保存至 {output_file}")

def save_pairs_to_csv(pairs, output_file):
    dir_path = os.path.dirname(output_file)
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    df = pd.DataFrame(pairs, columns=["Last Frame", "Current Frame", "Distance", "Size Change"])
    df['Face Index'] = df.groupby(['Last Frame', 'Current Frame']).cumcount()
    df = df[["Last Frame", "Current Frame", "Face Index", "Distance"]]
    csv_output_file = output_file if output_file.endswith('.csv') else output_file + '.csv'
    df.to_csv(csv_output_file, index=False)
    
    print(f"frame distance已保存至 {csv_output_file}")


def create_video_from_frames(start_frame, end_frame, output_video_path, input_frames_dir, frame_rate=30, bit_rate='200'):
    input_pattern = os.path.join(input_frames_dir, f"%06d.png")
    command = [
        '/opt/conda/envs/work38/bin/ffmpeg',
        '-y',
        '-framerate', str(frame_rate),
        '-start_number', str(start_frame),
        '-i', input_pattern,
        '-vframes', str(end_frame - start_frame + 1),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-bit_rate', bit_rate,
        output_video_path
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"创建视频失败: {e}")

def process_clips(annotations_file, input_frames_dir, input_emoca_dir, output_videos_dir,
                  frame_rate=30, bit_rate='200', sorted_sizes=None, sorted_centers=None):
    if not os.path.exists(output_videos_dir):
        os.makedirs(output_videos_dir)

    if Path(input_frames_dir).is_file():
        file_dir = Path(output_videos_dir).parent
        out_folder = file_dir / 'frames'
        out_folder.mkdir(parents=True, exist_ok=True)
        print("正在解压视频到 '%s'" % str(out_folder))

        out_format = out_folder / ("%06d" + ".jpg")
        out_format = '-r 1 -i %s -r 1 ' % str(input_frames_dir) + ' "' + str(out_format) + '"'

        os.system("/opt/conda/envs/work38/bin/ffmpeg " + out_format)
        input_frames_dir = out_folder

    if Path(input_emoca_dir).is_file():
        file_dir = Path(output_videos_dir).parent
        out_folder = file_dir / 'emoca_frames'
        out_folder.mkdir(parents=True, exist_ok=True)
        print("正在解压视频到 '%s'" % str(out_folder))

        out_format = out_folder / ("%06d" + ".png")
        out_format = '-r 1 -i %s -r 1 ' % str(input_emoca_dir) + ' "' + str(out_format) + '"'

        os.system("/opt/conda/envs/work38/bin/ffmpeg " + out_format)
        input_emoca_dir = out_folder

    with open(annotations_file, 'r') as file:
        clips = file.readlines()

    for i, clip in enumerate(clips):
        start_frame, end_frame = map(int, clip.strip().split())
        clip_length = end_frame - start_frame + 1

        if clip_length >= 3 * frame_rate:
            output_video_path = os.path.join(output_videos_dir, f"clip_{start_frame}-{end_frame}.mp4")
            create_video_from_frames(start_frame, end_frame, output_video_path, input_frames_dir, frame_rate, bit_rate)
            output_video_path_emoca = os.path.join(output_videos_dir, f"emoca_{start_frame}-{end_frame}.mp4")
            create_video_from_frames(start_frame, end_frame, output_video_path_emoca, input_emoca_dir, frame_rate, bit_rate)
            # 只有在成功创建视频时才保存bbox数据
            if sorted_sizes is not None and sorted_centers is not None:
                bbox_output_dir = os.path.join(output_videos_dir, 'bbox')
                if not os.path.exists(bbox_output_dir):
                    os.makedirs(bbox_output_dir)
                
                bbox_output_file = os.path.join(bbox_output_dir, f'clip_{start_frame}-{end_frame}_bbox.pkl')
                clip_sizes = sorted_sizes[start_frame-1:end_frame]
                clip_centers = sorted_centers[start_frame-1:end_frame]
                
                with open(bbox_output_file, 'wb') as f:
                    pkl.dump(clip_sizes, f)
                    pkl.dump(clip_centers, f)
                print(f"已保存bbox数据到 {bbox_output_file}")

    print(f"视频已保存到目录: {output_videos_dir}")

# 主程序
import concurrent.futures
import os

def process_video(video_dir, video_name, output_dir):
    results_dir = f'{video_dir}/{video_name}/results'
    
    # 检查是否已经处理过
    # if os.path.exists(f'/amax/zyude/results_10_6/{videos_list}/{video_name}') or \
    #    os.path.exists(f'/amax/zyude/data/results_10_2/{videos_list}/{video_name}'):
    #     print(f"跳过已处理的视频: {video_name}")
    #     return

    print(f"正在处理: {results_dir}")  # 输出正在处理的video_dir

    for T in [0.02]:
        output_file = os.path.join(output_dir, f'clip_annotations_{T}.txt')
    
        with open(f"{video_dir}/{video_name}/{video_name}/detections/bboxes.pkl", "rb") as f:
            detection_fnames = pkl.load(f)
            centers = pkl.load(f)
            sizes = pkl.load(f)

        clips, frame_pair, sorted_sizes, sorted_centers = detect_shot_changes(results_dir, sizes, centers, T)
        save_clips_to_file(clips, output_file)

        annotations_file = output_file
        input_frames_dir = f'{video_dir}/{video_name}/{video_name}/videos'
        input_emoca_dir = f'{video_dir}/{video_name}/results/video_geometry_coarse.mp4'
        output_videos_dir = f'{output_dir}/clips'

        with open(f'{video_dir}/{video_name}/metadata.pkl', "rb") as f:
            _ = pkl.load(f)
            _ = pkl.load(f)
            info = pkl.load(f)
            info = info[0]
        
        fps = int(info['fps'].split('/')[0]) / int(info['fps'].split('/')[1])
        bit_rate = info['bit_rate']

        process_clips(annotations_file, input_frames_dir, input_emoca_dir, output_videos_dir, int(fps), bit_rate, sorted_sizes, sorted_centers)

        save_pairs_to_csv(frame_pair, os.path.join(output_dir, 'frame_distance.csv'))

    # 清理内存缓存
    gc.collect()

# 遍历/amax/data/human_videos/results/bilibili下的子目录
root_dir = '/zouyude/data/bilibili_test'
output_base_dir = '/zouyude/clip_results'
max_workers = 8  # 设置最大并发数为8

# 先收集所有需要处理的视频
videos_to_process = []
for videos_list in os.listdir(root_dir):
    video_dir = os.path.join(root_dir, videos_list, 'EMOCA_v2_lr_mse_20')
    if os.path.isdir(video_dir):
        video_list = [item for item in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, item))]
        for video_name in video_list:
            output_dir = os.path.join(output_base_dir, videos_list, video_name)
            if os.path.exists(os.path.join(output_dir, 'frame_distance.csv')):
                continue
            if not os.path.exists(f'{video_dir}/{video_name}/results/video_geometry_coarse.mp4'):
                continue
            videos_to_process.append((video_dir, video_name, output_dir))

print(f"共有{len(videos_to_process)}个视频需要处理")

# with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
#     futures = []
#     for video_dir, video_name, output_dir in tqdm(videos_to_process, desc="提交处理任务"):
#         futures.append(executor.submit(process_video, video_dir, video_name, output_dir))
    
#     for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理进度"):
#         future.result()

# import cProfile
# import pstats
# profiler = cProfile.Profile()
# profiler.enable()

for video_dir, video_name, output_dir in tqdm(videos_to_process, desc="处理视频"):
    process_video(video_dir, video_name, output_dir)

# profiler.disable()
# # 输出性能报告
# with open("clip_video_track_face.txt", "w") as f:
#     stats = pstats.Stats(profiler, stream=f)
#     stats.strip_dirs()
#     stats.sort_stats("cumtime")  # 可以根据其他统计方式排序，如 'time', 'ncalls'
#     stats.print_stats()

# for videos_list in os.listdir(root_dir):
#     video_dir = os.path.join(root_dir, videos_list, 'EMOCA_v2_lr_mse_20')
#     if os.path.isdir(video_dir):
#         video_list = [item for item in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, item))]
#         for video_name in video_list:
#             output_dir = os.path.join(output_base_dir, videos_list, video_name)
#             process_video(video_dir, video_name, output_dir)

print("所有视频处理完成")