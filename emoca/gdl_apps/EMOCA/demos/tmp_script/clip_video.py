import os
import subprocess
from pathlib import Path
from tqdm import tqdm

def calculate_centroid(obj_file_path):
    vertices = []
    try:
        with open(obj_file_path, 'r') as obj_file:
            for line in obj_file:
                if line.startswith('v '):
                    parts = line.split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append((x, y, z))
        if vertices:
            avg_x = sum(v[0] for v in vertices) / len(vertices)
            avg_y = sum(v[1] for v in vertices) / len(vertices)
            avg_z = sum(v[2] for v in vertices) / len(vertices)
            return avg_x, avg_y, avg_z
    except FileNotFoundError:
        # 如果文件不存在，返回 None
        return None

def calculate_distance(centroid1, centroid2):
    if centroid1 and centroid2:
        diff_x = abs(centroid1[0] - centroid2[0])
        diff_y = abs(centroid1[1] - centroid2[1])
        diff_z = abs(centroid1[2] - centroid2[2])
        return (diff_x + diff_y + diff_z)/3  # 返回质心的总差值
    return None

def detect_shot_changes(results_dir, threshold=0.02):
    # 列出所有帧目录（包含不同人脸），按照帧号和人脸编号排序
    frame_dirs = sorted([d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))])

    clips = []
    current_clip_start = None
    last_centroids = {}  # 保存上一帧所有人脸的质心 {帧号: [质心1, 质心2, ...]}
    last_faces_count = 0  # 上一帧的人脸数量
    last_frame_number = None  # 保存前一帧的帧号

    current_frame_faces = []  # 保存当前帧的人脸质心
    current_frame_number = None  # 保存当前帧的帧号

    for frame_dir in tqdm(frame_dirs):
        frame_number, face_id = frame_dir.split('_')  # 获取帧号和人脸编号
        face_id = int(face_id)

        obj_file = os.path.join(results_dir, frame_dir, 'mesh_coarse.obj')
        current_centroid = calculate_centroid(obj_file)

        # 保存当前帧的质心
        if current_centroid:
            if current_frame_number is None:
                current_frame_number = frame_number  # 设定当前帧号

            # 判断是否处理的是同一帧的其他人脸
            if frame_number == current_frame_number:
                current_frame_faces.append(current_centroid)
            else:
                # 当前帧处理完了，开始进行切片判断

                # 帧号不连续，切片
                if last_frame_number and int(current_frame_number) != int(last_frame_number) + 1:
                    if current_clip_start is not None:
                        clips.append((current_clip_start, last_frame_number))
                    current_clip_start = current_frame_number
                    last_centroids[current_frame_number] = current_frame_faces[:]
                    last_faces_count = len(current_frame_faces)
                    last_frame_number = current_frame_number
                    # 跳过其他判断，直接处理下一帧
                    current_frame_faces = [current_centroid]
                    current_frame_number = frame_number
                    continue

                # 人脸数量变化，切片
                if last_faces_count != len(current_frame_faces):
                    if current_clip_start is not None:
                        clips.append((current_clip_start, last_frame_number))
                    current_clip_start = current_frame_number
                    last_centroids[current_frame_number] = current_frame_faces[:]
                    last_faces_count = len(current_frame_faces)
                    last_frame_number = current_frame_number
                    # 跳过其他判断，直接处理下一帧
                    current_frame_faces = [current_centroid]
                    current_frame_number = frame_number
                    continue

                # 质心差异判断
                if last_faces_count > 0:
                    for i in range(min(len(current_frame_faces), last_faces_count)):
                        distance = calculate_distance(last_centroids[last_frame_number][i], current_frame_faces[i])
                        if distance > threshold:
                            # 质心变化超过阈值，切片
                            if current_clip_start is not None:
                                clips.append((current_clip_start, last_frame_number))
                            current_clip_start = current_frame_number
                            break

                # 更新上一帧数据
                last_centroids[current_frame_number] = current_frame_faces[:]
                last_faces_count = len(current_frame_faces)
                last_frame_number = current_frame_number

                # 重置当前帧数据
                current_frame_faces = [current_centroid]
                current_frame_number = frame_number

    # 处理最后一段
    if current_clip_start is not None and last_frame_number and int(current_clip_start) <= int(last_frame_number):
        clips.append((current_clip_start, last_frame_number))

    return clips

def save_clips_to_file(clips, output_file):
    with open(output_file, 'w') as file:
        for clip in clips:
            file.write(f"{clip[0]} {clip[1]}\n")

def create_video_from_frames(start_frame, end_frame, output_video_path, input_frames_dir, frame_rate=30):
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
        output_video_path  # 输出视频文件路径
    ]

    # 执行 ffmpeg 命令
    subprocess.run(command, check=True)

def process_clips(annotations_file, input_frames_dir, output_videos_dir, frame_rate=30):
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_videos_dir):
        os.makedirs(output_videos_dir)

    if Path(input_frames_dir).is_file():
        file_dir = Path(input_frames_dir).parent.parent
        out_folder = file_dir / 're_videos'
        out_folder.mkdir(parents=True, exist_ok=True)
        print("Unpacking video to '%s'" % str(out_folder))

        out_format = out_folder / ("%06d" + ".png")
        out_format = '-r 1 -i %s -r 1 ' % str(input_frames_dir) + ' "' + str(out_format) + '"'
        # out_format = ' -r 1 -i %s ' % str(video_file) + ' "' + "$frame.%03d.png" + '"'
        # subprocess.call(['ffmpeg', out_format])
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

        if clip_length >= frame_rate:
            # 输出视频文件路径
            output_video_path = os.path.join(output_videos_dir, f"clip_{i+1:03d}.mp4")

            # 调用 ffmpeg 生成视频
            create_video_from_frames(start_frame, end_frame, output_video_path, input_frames_dir, frame_rate)

    print(f"视频已保存到目录: {output_videos_dir}")


# 主程序
video_list = ['BV162421F7sk']
# BV165411x7nA.mp4
for video_name in video_list:
    # video_name = 'BV11R4y1U7br'
    results_dir = f'/mnt/d/emoca/video_output/EMOCA_v2_lr_mse_20/{video_name}/results'
    output_file = os.path.join(Path(results_dir).parent, 'clip_annotations.txt')

    if os.path.exists(output_file):
        print(f"剪辑注释 {output_file} 已存在。")
    else:
        # 检测镜头切换并保存注释
        clips = detect_shot_changes(results_dir)
        save_clips_to_file(clips, output_file)

        print(f"剪辑注释已保存至 {output_file}")


    # 主程序
    annotations_file = output_file
    input_frames_dir = f'{results_dir}/video_geometry_detail.mp4'
    output_videos_dir = f'/mnt/d/emoca/clips/{video_name}'

    # fps = int(self.video_metas[sequence_id]['fps'].split('/')[0]) / int(self.video_metas[sequence_id]['fps'].split('/')[1])

    # 处理剪辑并生成视频
    process_clips(annotations_file, input_frames_dir, output_videos_dir)