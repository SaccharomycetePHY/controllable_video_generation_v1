"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""
import cProfile
import pstats
import os
import copy
import multiprocessing
from multiprocessing import Process, JoinableQueue, Queue
import queue
import numpy as np
import gdl.utils.DecaUtils as util
from gdl.utils.lightning_logging import _fix_image
from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
import gdl
from pathlib import Path
from tqdm import auto
import argparse
from gdl_apps.EMOCA.utils.io import test
from skimage.io import imsave
import psutil
import time
import gc
import pickle
import torch
from multiprocessing import Pool
import torch.multiprocessing as mp
from tqdm import tqdm
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def reconstruct_video(args, device):
    path_to_models = args.path_to_models
    input_video = args.input_video
    model_name = args.model_name
    output_folder = args.output_folder + "/" + model_name
    image_type = args.image_type
    black_background = args.black_background
    include_original = args.include_original
    include_rec = args.include_rec
    cat_dim = args.cat_dim
    use_mask = args.use_mask
    include_transparent = bool(args.include_transparent)
    processed_subfolder = args.processed_subfolder

    mode = args.mode
    # mode = 'detail'
    # mode = 'coarse'

    start_time = time.time()
    print("正在处理视频：", input_video)

    ## 1) Process the video - extract the frames from video and detected faces
    # processed_subfolder="processed_2022_Jan_15_02-43-06"
    processed_subfolder=Path(input_video).stem
    
    dm = TestFaceVideoDM(input_video, output_folder, processed_subfolder=processed_subfolder, 
        batch_size=32, num_workers=0, device=device)
    dm.prepare_data()
    dm.setup()
    processed_subfolder = Path(dm.output_dir).name

    # ## 2) Load the model
    emoca, conf = load_model(path_to_models, model_name, mode)
    emoca.to(device)
    emoca.eval()

    if Path(output_folder).is_absolute():
        outfolder = output_folder
    else:
        outfolder = str(Path(output_folder) / processed_subfolder / Path(input_video).stem / "results" / model_name)
    
    outfolder = str(Path(output_folder) / processed_subfolder / "results")
    ## 3) Get the data loadeer with the detected faces
    dl = dm.test_dataloader()

    # # Create a queue for saving data
    # save_queue = Queue()  # Adjust maxsize as needed

    # # Start the worker process
    # NUM_WORKERS = 1
    # workers = []
    # for _ in range(NUM_WORKERS):
    #     p = Process(target=save_worker_func, args=(save_queue,))
    #     p.start()
    #     workers.append(p)

    # ## 4) 在数据上运行模型
    # 主循环
    # 设置内存使用阈值（例如80%）
    MEM_THRESHOLD = 80

    for j, batch in enumerate(auto.tqdm(dl)):
        current_bs = batch["image"].shape[0]
        img = batch
        vals, visdict = test(emoca, img)
        
        # # 将vals和visdict保存为文件
        # for i in range(current_bs):
        #     name = batch["image_name"][i]
        #     sample_output_folder = Path(outfolder) / name
        #     sample_output_folder.mkdir(parents=True, exist_ok=True)

            # for k in vals:
            #     print(k, vals[k].type())

            # 保存vals
            # vals_path = sample_output_folder / f"{name}_vals.pkl"
            # with open(vals_path, 'wb') as f:
            #     pickle.dump({k: vals[k][i].detach().cpu() for k in ['shapecode', 'texcode', 'expcode', 'posecode', 'cam', 'lightcode'] if k in vals and isinstance(vals[k], torch.cuda.FloatTensor)}, f)
            
            # # 保存visdict
            # visdict_path = sample_output_folder / f"{name}_visdict.pkl"
            # with open(visdict_path, 'wb') as f:
            #     pickle.dump({k: visdict[k][i].detach().cpu() for k in visdict if isinstance(visdict[k], torch.cuda.FloatTensor)}, f)

        # 提取保存所需的数据，移至CPU
        faces = emoca.deca.render.faces[0].detach().cpu().numpy()
        uvcoords = emoca.deca.render.raw_uvcoords[0].detach().cpu().numpy()
        uvfaces = emoca.deca.render.uvfaces[0].detach().cpu().numpy()

        for i in range(current_bs):
            name = batch["image_name"][i]
            sample_output_folder = Path(outfolder) / name
            sample_output_folder.mkdir(parents=True, exist_ok=True)

            # 准备save_obj的数据
            if args.save_mesh:
                opdict = {
                    'verts': vals['verts'][i].detach().cpu().numpy(),
                    'uv_texture_gt': vals['uv_texture_gt'][i].detach().cpu(),
                    'uv_detail_normals': vals['uv_detail_normals'][i].detach().cpu(),
                    'faces': faces,
                    'uvcoords': uvcoords,
                    'uvfaces': uvfaces,
                }
                save_obj(str(sample_output_folder / "mesh_coarse.obj"), opdict)

            # 准备save_images的数据
            if args.save_images:
                visdict_i = {key: visdict[key][i].detach().cpu() for key in visdict}
                save_images(outfolder, name, visdict_i)

            # 准备save_codes的数据
            if args.save_codes:
                codes = {
                    'shapecode': vals['shapecode'][i].detach().cpu().numpy(),
                    'expcode': vals['expcode'][i].detach().cpu().numpy(),
                    'texcode': vals['texcode'][i].detach().cpu().numpy(),
                    'posecode': vals['posecode'][i].detach().cpu().numpy(),
                    'detailcode': vals['detailcode'][i].detach().cpu().numpy(),
                    'cam': vals['cam'][i].detach().cpu().numpy(),
                }
                save_codes(Path(outfolder), name, codes)

        # # 检查内存使用情况
        # mem_percent = psutil.virtual_memory().percent
        # if mem_percent > MEM_THRESHOLD:
        #     print(f"内存使用率达到{mem_percent}%，开始处理队列任务...")
        #     while mem_percent > MEM_THRESHOLD and not save_queue.empty():
        #         try:
        #             item = save_queue.get_nowait()
        #             if item != 'STOP':
        #                 func_name, *args = item
        #                 if func_name == 'save_obj':
        #                     save_obj(*args)
        #                 elif func_name == 'save_images':
        #                     save_images(*args)
        #                 elif func_name == 'save_codes':
        #                     save_codes(*args)
        #             save_queue.task_done()
                    
        #             # 处理完一个任务后立即检查内存使用率
        #             mem_percent = psutil.virtual_memory().percent
        #             if mem_percent <= MEM_THRESHOLD:
        #                 print(f"内存使用率降低到{mem_percent}%，恢复主要处理流程...")
        #                 break
        #         except queue.Empty:
        #             break

        #     if mem_percent > MEM_THRESHOLD:
        #         print(f"队列处理完毕，但内存使用率仍为{mem_percent}%，可能需要进一步优化...")
        #     else:
        #         print(f"内存使用率降低到{mem_percent}%，继续处理...")
        
        # 清理内存
        del vals, visdict, faces, uvcoords, uvfaces
        gc.collect()


    # # 向工作线程发送退出信号
    # for _ in range(NUM_WORKERS):
    #     save_queue.put('STOP')

    # # 等待所有工作进程结束
    # for p in workers:
    #     p.join()

    if args.reconstruct_video and args.save_images:
        video_file, video_file_with_sound = dm.create_reconstruction_video(0,  rec_method=model_name, image_type=image_type, overwrite=True, 
                cat_dim=cat_dim, include_transparent=include_transparent, 
                include_original=include_original, 
                include_rec = include_rec,
                black_background=black_background, 
                use_mask=use_mask, 
                out_folder=outfolder)
        print("视频保存到：", video_file, "，用时：", time.time() - start_time)

    print("Done")


def save_obj(filename, opdict):
    dense_template_path = Path(gdl.__file__).parents[1] / 'assets' / "DECA" / "data" / 'texture_data_256.npy'
    dense_template = np.load(dense_template_path, allow_pickle=True, encoding='latin1').item()
    vertices = opdict['verts']
    faces = opdict['faces']
    texture = util.tensor2image(opdict['uv_texture_gt'])
    uvcoords = opdict['uvcoords']
    uvfaces = opdict['uvfaces']
    normal_map = util.tensor2image(opdict['uv_detail_normals'] * 0.5 + 0.5)
    util.write_obj(filename, vertices, faces,
                   texture=texture,
                   uvcoords=uvcoords,
                   uvfaces=uvfaces,
                   normal_map=normal_map)
    # 释放内存
    del dense_template, vertices, faces, texture, uvcoords, uvfaces, normal_map

def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(1, 2, 0)

def save_images(outfolder, name, vis_dict):
    final_out_folder = Path(outfolder) / name
    final_out_folder.mkdir(parents=True, exist_ok=True)
    imsave(final_out_folder / "geometry_coarse.png", _fix_image(torch_img_to_np(vis_dict['geometry_coarse'])))
    # imsave(final_out_folder / "geometry_detail.png", _fix_image(torch_img_to_np(vis_dict['geometry_detail'])))
    # imsave(final_out_folder / "out_im_coarse.png", _fix_image(torch_img_to_np(vis_dict['output_images_coarse'])))
    # imsave(final_out_folder / "out_im_detail.png", _fix_image(torch_img_to_np(vis_dict['output_images_detail'])))
    # 释放内存
    del vis_dict

def save_codes(output_folder, name, codes):
    np.save(output_folder / name / "shape.npy", codes["shapecode"])
    np.save(output_folder / name / "exp.npy", codes["expcode"])
    np.save(output_folder / name / "tex.npy", codes["texcode"])
    np.save(output_folder / name / "pose.npy", codes["posecode"])
    np.save(output_folder / name / "detail.npy", codes["detailcode"])
    np.save(output_folder / name / "cam.npy", codes["cam"])
    # 释放内存
    del codes

def reconstruct_video_one_gpu(args, input_videos, output_videos, gpu_id):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    for idx, video in tqdm(enumerate(input_videos), total=len(input_videos), desc=f"GPU{gpu_id}处理视频"):
        print(f"GPU{gpu_id}共有视频{len(input_videos)}, 正在处理 {idx+1}")
        args_copy = copy.deepcopy(args)
        args_copy.input_video = video
        args_copy.output_folder = output_videos[idx]
        try:
            reconstruct_video(args_copy, device)
        except Exception as e:
            print(f"处理视频{video}时发生错误: {e}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_input_path', required=True, type=str)
    parser.add_argument('--base_output_path', required=True, type=str)
    parser.add_argument('--input_video', type=str, default="/amax/zyude/SMPLer-X/demo/videos/BV14z421R7mc.mp4", 
    # parser.add_argument('--input_video', type=str, default=str(Path(gdl.__file__).parents[1] / "data/EMOCA_test_example_data/videos/bilibili.mp4"), 
        help="Filename of the video for reconstruction.")
    parser.add_argument('--output_folder', type=str, default="video_output", help="Output folder to save the results to.")
    parser.add_argument('--reconstruct_video', type=str2bool, default=True, help="If true, must save images and will render the video")
    parser.add_argument('--model_name', type=str, default='EMOCA_v2_lr_mse_20', help='Name of the model to use. Currently EMOCA or DECA are available.')
    parser.add_argument('--path_to_models', type=str, default=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models"))
    parser.add_argument('--mode', type=str, default="detail", choices=["detail", "coarse"], help="Which model to use for the reconstruction.")
    parser.add_argument('--save_images', type=str2bool, default=True, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=str2bool, default=True, help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=str2bool, default=False, help="If true, output meshes will be saved")
    # add a string argument with several options for image type
    parser.add_argument('--image_type', type=str, default='geometry_coarse', 
        choices=["geometry_detail", "geometry_coarse", "out_im_detail", "out_im_coarse"], 
        help="Which image to use for the reconstruction video.")
    parser.add_argument('--processed_subfolder', type=str, default=None, 
        help="If you want to resume previously interrupted computation over a video, make sure you specify" \
            "the subfolder where the got unpacked. It will be in format 'processed_%Y_%b_%d_%H-%M-%S'")
    parser.add_argument('--cat_dim', type=int, default=1, 
        help="The result video will be concatenated vertically if 0 and horizontally if 1")
    parser.add_argument('--include_rec', type=str2bool, default=True, 
        help="The reconstruction (non-transparent) will be in the video if True")
    parser.add_argument('--include_transparent', type=str2bool, default=False, 
        help="Apart from the reconstruction video, also a video with the transparent mesh will be added")
    parser.add_argument('--include_original', type=str2bool, default=False, 
        help="Apart from the reconstruction video, also a video with the transparent mesh will be added")
    parser.add_argument('--black_background', type=str2bool, default=True, help="If true, the background of the reconstruction video will be black")
    parser.add_argument('--use_mask', type=str2bool, default=True, help="If true, the background of the reconstruction video will be black")
    parser.add_argument('--logger', type=str, default="", choices=["", "wandb"], help="Specify how to log the results if at all.")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    input_videos, output_videos = [], []
    base_input_path = args.base_input_path
    base_output_path = args.base_output_path

    videos_list = os.listdir(base_input_path)  # 获取所有视频列表

    # # 读取需要处理的视频列表
    # with open('/zouyude/data/unprocessed_videos.txt', 'r') as f:
    #     videos_to_process = [line.strip() for line in f.readlines()]

    for list_name in videos_list:
        # if list_name != '389988163':
        #     continue
        list_path = os.path.join(base_input_path, list_name)
        if os.path.isdir(list_path):
            for file in os.listdir(list_path):
                if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # 根据需要添加更多的扩展名
                    video_name = os.path.splitext(file)[0]
                    # # 如果视频名不在列表中则跳过
                    # if video_name not in videos_to_process:
                    #     continue
                    output_video_path = os.path.join(base_output_path, list_name, 'EMOCA_v2_lr_mse_20', video_name, 'results', 'video_geometry_coarse.mp4')
                    if not os.path.exists(output_video_path):
                        input_videos.append(os.path.join(list_path, file))
                        output_videos.append(os.path.join(base_output_path, list_name))

    print("一共有", len(input_videos), "个视频")

    # 多进程处理
    num_gpus = torch.cuda.device_count()
    videos_per_gpu = len(input_videos) // num_gpus
    processes = []
    
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * videos_per_gpu
        end_idx = start_idx + videos_per_gpu if gpu_id < num_gpus - 1 else len(input_videos)
        gpu_videos = input_videos[start_idx:end_idx]
        gpu_output_videos = output_videos[start_idx:end_idx]
        p = mp.Process(target=reconstruct_video_one_gpu, args=(args, gpu_videos, gpu_output_videos, gpu_id))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
