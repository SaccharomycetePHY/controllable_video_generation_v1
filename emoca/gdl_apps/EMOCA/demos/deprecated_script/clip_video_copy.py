import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import pandas as pd
def read_vertices(obj_file_path):
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
        # 如果文件不存在，返回 None
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

def detect_shot_changes(results_dir, sizes, threshold=0.02):
    # 列出所有帧目录（包含不同人脸），按照帧号和人脸编号排序
    frame_dirs = sorted([d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))])

    frame_pair = []
    clips = []
    current_clip_start = None
    last_vertices_dict = {}  # 保存上一帧所有人脸的顶点数据 {帧号: [vertices1, vertices2, ...]}
    last_faces_count = 0  # 上一帧的人脸数量
    last_frame_number = None  # 保存前一帧的帧号

    current_frame_vertices_list = []  # 保存当前帧的人脸顶点列表
    current_frame_number = None  # 保存当前帧的帧号

    for frame_dir in tqdm(frame_dirs):
        frame_number, face_id = frame_dir.split('_')  # 获取帧号和人脸编号
        face_id = int(face_id)
        
        if sizes[int(frame_number)-1][face_id] <= 80:    # 过滤小人脸
            continue

        obj_file = os.path.join(results_dir, frame_dir, 'mesh_coarse.obj')
        current_vertices = read_vertices(obj_file)

        # 保存当前帧的顶点数据
        if current_vertices:
            if current_frame_number is None:
                current_frame_number = frame_number  # 设定当前帧号

            # 判断是否处理的是同一帧的其他人脸
            if frame_number == current_frame_number:
                current_frame_vertices_list.append(current_vertices)
            else:
                # 当前帧处理完了，开始进行切片判断

                # 帧号不连续，切片
                if last_frame_number and int(current_frame_number) != int(last_frame_number) + 1:
                    if current_clip_start is not None:
                        clips.append((current_clip_start, last_frame_number))
                    current_clip_start = current_frame_number
                    last_vertices_dict[current_frame_number] = current_frame_vertices_list[:]
                    last_faces_count = len(current_frame_vertices_list)
                    last_frame_number = current_frame_number
                    # 跳过其他判断，直接处理下一帧
                    current_frame_vertices_list = [current_vertices]
                    current_frame_number = frame_number
                    continue

                # 人脸数量变化，切片
                if last_faces_count != len(current_frame_vertices_list):
                    if current_clip_start is not None:
                        clips.append((current_clip_start, last_frame_number))
                    current_clip_start = current_frame_number
                    last_vertices_dict[current_frame_number] = current_frame_vertices_list[:]
                    last_faces_count = len(current_frame_vertices_list)
                    last_frame_number = current_frame_number
                    # 跳过其他判断，直接处理下一帧
                    current_frame_vertices_list = [current_vertices]
                    current_frame_number = frame_number
                    continue

                # 质心差异判断，改为顶点平均误差判断
                if last_faces_count > 0 and len(current_frame_vertices_list) > 0:
                    min_faces_count = min(last_faces_count, len(current_frame_vertices_list))
                    max_faces_count = max(last_faces_count, len(current_frame_vertices_list))
                    
                    for i in range(min_faces_count):
                        distance = []
                        size_change = []
                        for j in range(max_faces_count):
                            if last_faces_count > len(current_frame_vertices_list):
                                distance.append(calculate_max_distance(last_vertices_dict[last_frame_number][j], current_frame_vertices_list[i]))
                                size_change.append(abs(sizes[int(last_frame_number)-1][j] - sizes[int(current_frame_number)-1][i]))
                            else:
                                distance.append(calculate_max_distance(last_vertices_dict[last_frame_number][i], current_frame_vertices_list[j]))
                                size_change.append(abs(sizes[int(last_frame_number)-1][i] - sizes[int(current_frame_number)-1][j]))
                        
                        min_distance = min(distance)
                        min_size_change = min(size_change)
                        frame_pair.append((last_frame_number, current_frame_number, min_distance, min_size_change))
                        
                        if min_distance > threshold or min_size_change > 30:
                            # 平均误差超过阈值或尺寸变化超过30，切片
                            if current_clip_start is not None:
                                clips.append((current_clip_start, last_frame_number))
                            current_clip_start = current_frame_number
                            break

                # 更新上一帧数据
                last_vertices_dict[current_frame_number] = current_frame_vertices_list[:]
                last_faces_count = len(current_frame_vertices_list)
                last_frame_number = current_frame_number

                # 重置当前帧数据
                current_frame_vertices_list = [current_vertices]
                current_frame_number = frame_number

    # 处理最后一段
    if current_clip_start is not None and last_frame_number and int(current_clip_start) <= int(last_frame_number):
        clips.append((current_clip_start, last_frame_number))

    return clips, frame_pair

def save_clips_to_file(clips, output_file):
    # 检查目录是否存在，不存在则创建
    if not os.path.exists(Path(output_file).parent):
        os.makedirs(Path(output_file).parent)

    with open(output_file, 'w') as file:
        for clip in clips:
            file.write(f"{clip[0]} {clip[1]}\n")
    print(f"剪辑注释已保存至 {output_file}")

def save_pairs_to_csv(pairs, output_file):
    # 获取文件所在目录的路径
    dir_path = os.path.dirname(output_file)
    
    # 检查目录是否存在，不存在则创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # 将 pairs 数据转换为 pandas DataFrame
    df = pd.DataFrame(pairs, columns=["Last Frame", "Current Frame", "Distance", "Size Change"])

    # 通过 "Last Frame" 和 "Current Frame" 对行进行分组，并为每个组添加 "Face Index"
    df['Face Index'] = df.groupby(['Last Frame', 'Current Frame']).cumcount()

    # 调整列的顺序，把 "Face Index" 放在 "Distance" 前面
    df = df[["Last Frame", "Current Frame", "Face Index", "Distance"]]

    # 保存为 CSV 文件
    csv_output_file = output_file if output_file.endswith('.csv') else output_file + '.csv'
    df.to_csv(csv_output_file, index=False)
    
    print(f"frame distance已保存至 {csv_output_file}")

def create_video_from_frames(start_frame, end_frame, output_video_path, input_frames_dir, frame_rate=30, bit_rate='200'):
    # 构建图片的输入格式，文件名格式为 %06d.png (例如: 000001.png)
    input_pattern = os.path.join(input_frames_dir, f"%06d.png")

    # 通过 ffmpeg 将图片转为视频
    command = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-framerate', str(frame_rate),  # 设置帧率
        '-start_number', str(start_frame),  # 起始帧
        '-i', input_pattern,  # 输入图片的格式
        '-vframes', str(end_frame - start_frame + 1),  # 处理的帧数
        '-c:v', 'libx264',  # 编码器
        '-pix_fmt', 'yuv420p',  # 像素格式
        '-bit_rate', bit_rate,
        output_video_path  # 输出视频文件路径
    ]

    # 执行 ffmpeg 命令
    subprocess.run(command, check=True)

def process_clips(annotations_file, input_frames_dir, output_videos_dir, frame_rate=30, bit_rate='200'):
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_videos_dir):
        os.makedirs(output_videos_dir)

    if Path(input_frames_dir).is_file():
        file_dir = Path(output_videos_dir).parent
        out_folder = file_dir / 'frames'
        out_folder.mkdir(parents=True, exist_ok=True)
        print("Unpacking video to '%s'" % str(out_folder))

        out_format = out_folder / ("%06d" + ".jpg")  # 将输出格式改为 .jpg
        out_format = '-r 1 -i %s -r 1 ' % str(input_frames_dir) + ' "' + str(out_format) + '"'

        os.system("ffmpeg " + out_format)
        input_frames_dir = out_folder

    # 读取 clip_annotations.txt 文件
    with open(annotations_file, 'r') as file:
        clips = file.readlines()

    for i, clip in enumerate(clips):
        # 解析每一行，获取起始帧和结束帧
        start_frame, end_frame = map(int, clip.strip().split())

        # 计算clip的长度
        clip_length = end_frame - start_frame + 1

        if clip_length >= 3 * frame_rate:
            # 输出视频文件路径
            output_video_path = os.path.join(output_videos_dir, f"clip_{start_frame}-{end_frame}.mp4")

            # 调用 ffmpeg 生成视频
            create_video_from_frames(start_frame, end_frame, output_video_path, input_frames_dir, frame_rate, bit_rate)

    print(f"视频已保存到目录: {output_videos_dir}")


# 主程序
videos_list = '562197' # '141248' 146979897 1462401621 562197
video_dir = f'/amax/data/human_videos/results/bilibili/{videos_list}/EMOCA_v2_lr_mse_20' #'/mnt/d/emoca/video_output2/EMOCA_v2_lr_mse_20'
# 过滤出文件夹
video_list = [item for item in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, item))]
# video_list = ['BV15w4m1i7LH']
# BV165411x7nA.mp4
for video_name in video_list:
    # video_name = 'BV11R4y1U7br'
    results_dir = f'{video_dir}/{video_name}/results'
    output_dir = f'/amax/zyude/results_9_27/{videos_list}/{video_name}/'
    for T in [0.02]:
        output_file = os.path.join(output_dir, f'clip_annotations_{T}.txt')
    
        with open(f"{video_dir}/{video_name}/{video_name}/detections/bboxes.pkl", "rb" ) as f:
            detection_fnames = pkl.load(f)
            centers = pkl.load(f)
            sizes = pkl.load(f)

        # if os.path.exists(output_file):
        #     print(f"剪辑注释 {output_file} 已存在。")
        #     # os.remove(output_file)
        # else:
        #     # 检测镜头切换并保存注释
        clips, frame_pair = detect_shot_changes(results_dir, sizes, T)
        save_clips_to_file(clips, output_file)
        save_pairs_to_csv(frame_pair, f'{output_dir}frame_distance.csv')

        # 主程序
        annotations_file = output_file
        input_frames_dir = f'{video_dir}/{video_name}/{video_name}/videos' # f'{results_dir}/video_geometry_detail.mp4'
        output_videos_dir = f'{output_dir}/clips'

        with open(f'{video_dir}/{video_name}/metadata.pkl', "rb" ) as f:
            _ = pkl.load(f)
            _ = pkl.load(f)
            info = pkl.load(f)
            info = info[0]
        
        fps = int(info['fps'].split('/')[0]) / int(info['fps'].split('/')[1])
        bit_rate = info['bit_rate']

        # 处理剪辑并生成视频
        process_clips(annotations_file, input_frames_dir, output_videos_dir, int(fps), bit_rate)