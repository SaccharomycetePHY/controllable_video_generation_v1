import sys
sys.path.insert(0, '/amax/zyude/controllable_video_generation/emoca')
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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def reconstruct_video(args):
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
    processed_subfolder=Path(input_video).stem
    dm = TestFaceVideoDM(input_video, output_folder, processed_subfolder=processed_subfolder, 
        batch_size=32, num_workers=4)
    dm.prepare_data()
    dm.setup()
    processed_subfolder = Path(dm.output_dir).name
    outfolder = str(Path(output_folder) / processed_subfolder / "results")
    video_file, video_file_with_sound = dm.create_reconstruction_video(0,  rec_method=model_name, image_type=image_type, overwrite=True, 
            cat_dim=cat_dim, include_transparent=include_transparent, 
            include_original=include_original, 
            include_rec = include_rec,
            black_background=black_background, 
            use_mask=use_mask, 
            out_folder=outfolder)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use")
    parser.add_argument('--input_video', type=str, default="/amax/zyude/SMPLer-X/demo/videos/BV14z421R7mc.mp4", 
    # parser.add_argument('--input_video', type=str, default=str(Path(gdl.__file__).parents[1] / "data/EMOCA_test_example_data/videos/bilibili.mp4"), 
        help="Filename of the video for reconstruction.")
    parser.add_argument('--output_folder', type=str, default="video_output", help="Output folder to save the results to.")
    parser.add_argument('--reconstruct_video', type=str2bool, default=True, help="If true, must save images and will render the video")
    parser.add_argument('--model_name', type=str, default='EMOCA_v2_lr_mse_20', help='Name of the model to use. Currently EMOCA or DECA are available.')
    parser.add_argument('--path_to_models', type=str, default=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models"))
    parser.add_argument('--mode', type=str, default="detail", choices=["detail", "coarse"], help="Which model to use for the reconstruction.")
    parser.add_argument('--save_images', type=str2bool, default=True, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=str2bool, default=False, help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=str2bool, default=False, help="If true, output meshes will be saved")
    # add a string argument with several options for image type
    parser.add_argument('--image_type', type=str, default='geometry_detail', 
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

    import time
    
    start_time = time.time()

    input_videos = []
    args.input_video = '/amax/data/human_videos/bilibili/1462401621'    # 1462401621, 562197, 141248, 146979897
    args.output_folder = '/amax/data/human_videos/results/bilibili/1462401621'
    # args.output_folder = '/amax/zyude/controllable_video_generation/SMPLer-X/demo/results/clip_203-311_face'
    # input_videos = ['/amax/zyude/controllable_video_generation/SMPLer-X/demo/videos/clip_203-311.mp4']
    if os.path.isdir(args.input_video):
        for file in os.listdir(args.input_video):
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # 根据需要添加更多的扩展名
                input_videos.append(os.path.join(args.input_video, file))
    else:
        input_videos.append(args.input_video)
    print('共发现%d个视频' % len(input_videos))
    for idx, video in enumerate(input_videos):
        args_copy = copy.deepcopy(args)
        args_copy.input_video = video
        print('正在处理第%d个视频：%s' % (idx+1, video))
        reconstruct_video(args_copy)
        # 每处理完一个视频后清理内存
        gc.collect()

    end_time = time.time()
    print(f"主函数执行时间: {end_time - start_time:.2f} 秒")
    print("Done")


if __name__ == '__main__':
    main()
