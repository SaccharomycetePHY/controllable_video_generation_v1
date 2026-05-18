import subprocess
import threading
import os

# 定义执行shell脚本的函数
def run_inference(video_name):
    command = f"sh slurm_inference.sh {video_name} mp4 24 smpler_x_h32"
    print(f"正在处理视频: {video_name}")
    
    # 执行shell命令
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # 输出结果
    if process.returncode == 0:
        print(f"{video_name} 处理完成")
        print(stdout.decode())
    else:
        print(f"{video_name} 处理失败")
        print(stderr.decode())

# 提取目录中的所有视频文件，并去掉扩展名
def get_video_files(directory):
    # 获取所有文件，并过滤出mp4或其他你需要的格式，然后去掉文件的扩展名
    return [os.path.splitext(f)[0] for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.mp4')]

# 使用线程池批量处理所有视频文件
def process_videos_in_batches(video_files, process_num):
    threads = []
    
    # 逐个处理文件
    for video in video_files:
        # 如果当前线程数量达到process_num，等待其中一个完成
        while len(threads) >= process_num:
            for t in threads:
                if not t.is_alive():
                    threads.remove(t)
                    break
        
        # 创建并启动新线程
        t = threading.Thread(target=run_inference, args=(video,))
        threads.append(t)
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()

if __name__ == "__main__":
    # 设置并发执行的子线程数量
    process_num = 1
    
    # 提取视频文件（不带后缀）
    video_directory = "/amax/zyude/SMPLer-X/demo/results/"
    video_files = get_video_files(video_directory)
    
    # 确认获取到的视频文件列表
    print(f"发现 {len(video_files)} 个视频文件（不带后缀）: {video_files}")
    
    # 批量处理视频文件
    process_videos_in_batches(video_files, process_num)
