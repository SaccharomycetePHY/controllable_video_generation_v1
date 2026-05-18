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
    # mode = 'detail'
    # mode = 'coarse'
   
    ## 1) Process the video - extract the frames from video and detected faces
    # processed_subfolder="processed_2022_Jan_15_02-43-06"
    processed_subfolder=Path(input_video).stem
    dm = TestFaceVideoDM(input_video, output_folder, processed_subfolder=processed_subfolder, 
        batch_size=32, num_workers=4)
    dm.prepare_data()
    dm.setup()
    processed_subfolder = Path(dm.output_dir).name

    # ## 2) Load the model
    emoca, conf = load_model(path_to_models, model_name, mode)
    emoca.cuda()
    emoca.eval()

    if Path(output_folder).is_absolute():
        outfolder = output_folder
    else:
        outfolder = str(Path(output_folder) / processed_subfolder / Path(input_video).stem / "results" / model_name)
    
    outfolder = str(Path(output_folder) / processed_subfolder / "results")
    ## 3) Get the data loadeer with the detected faces
    dl = dm.test_dataloader()

    # Create a queue for saving data
    save_queue = Queue()  # Adjust maxsize as needed

    # Start the worker process
    NUM_WORKERS = 32
    workers = []
    for _ in range(NUM_WORKERS):
        p = Process(target=save_worker_func, args=(save_queue,))
        p.start()
        workers.append(p)

    # ## 4) Run the model on the data
    # Main loop
    for j, batch in enumerate(auto.tqdm(dl)):
        current_bs = batch["image"].shape[0]
        img = batch
        vals, visdict = test(emoca, img)

        # Extract data needed for saving, move to CPU
        faces = emoca.deca.render.faces[0].detach().cpu().numpy()
        uvcoords = emoca.deca.render.raw_uvcoords[0].detach().cpu().numpy()
        uvfaces = emoca.deca.render.uvfaces[0].detach().cpu().numpy()

        for i in range(current_bs):
            name = batch["image_name"][i]
            sample_output_folder = Path(outfolder) / name
            sample_output_folder.mkdir(parents=True, exist_ok=True)

            # Prepare data for save_obj
            if args.save_mesh:
                opdict = {
                    'verts': vals['verts'][i].detach().cpu().numpy(),
                    'uv_texture_gt': vals['uv_texture_gt'][i].detach().cpu(),
                    'uv_detail_normals': vals['uv_detail_normals'][i].detach().cpu(),
                    'faces': faces,
                    'uvcoords': uvcoords,
                    'uvfaces': uvfaces,
                }
                save_queue.put(('save_obj', str(sample_output_folder / "mesh_coarse.obj"), opdict))

            # Prepare data for save_images
            if args.save_images:
                visdict_i = {key: visdict[key][i].detach().cpu() for key in visdict}
                save_queue.put(('save_images', outfolder, name, visdict_i))

            # Prepare data for save_codes
            if args.save_codes:
                codes = {
                    'shapecode': vals['shapecode'][i].detach().cpu().numpy(),
                    'expcode': vals['expcode'][i].detach().cpu().numpy(),
                    'texcode': vals['texcode'][i].detach().cpu().numpy(),
                    'posecode': vals['posecode'][i].detach().cpu().numpy(),
                    'detailcode': vals['detailcode'][i].detach().cpu().numpy(),
                }
                save_queue.put(('save_codes', Path(outfolder), name, codes))

    # Signal the worker thread to exit
    for _ in range(NUM_WORKERS):
        save_queue.put('STOP')

    # Join worker processes
    for p in workers:
        p.join()

    print("Done")

def save_worker_func(queue):
    while True:
        try:
            item = queue.get()
            if item == 'STOP':
                # queue.task_done()
                break
            else:
                func_name, *args = item
                if func_name == 'save_obj':
                    save_obj(*args)
                elif func_name == 'save_images':
                    save_images(*args)
                elif func_name == 'save_codes':
                    save_codes(*args)
                # queue.task_done()
        except Exception as e:
            print(f"Exception in worker process: {e}")
            break
            # queue.task_done()


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

def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(1, 2, 0)

def save_images(outfolder, name, vis_dict):
    final_out_folder = Path(outfolder) / name
    final_out_folder.mkdir(parents=True, exist_ok=True)
    imsave(final_out_folder / "geometry_coarse.png", _fix_image(torch_img_to_np(vis_dict['geometry_coarse'])))
    imsave(final_out_folder / "geometry_detail.png", _fix_image(torch_img_to_np(vis_dict['geometry_detail'])))
    imsave(final_out_folder / "out_im_coarse.png", _fix_image(torch_img_to_np(vis_dict['output_images_coarse'])))
    imsave(final_out_folder / "out_im_detail.png", _fix_image(torch_img_to_np(vis_dict['output_images_detail'])))

def save_codes(output_folder, name, codes):
    np.save(output_folder / name / "shape.npy", codes["shapecode"])
    np.save(output_folder / name / "exp.npy", codes["expcode"])
    np.save(output_folder / name / "tex.npy", codes["texcode"])
    np.save(output_folder / name / "pose.npy", codes["posecode"])
    np.save(output_folder / name / "detail.npy", codes["detailcode"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, default='/amax/zyude/SMPLer-X/demo/videos/clip_203-311.mp4', 
    # parser.add_argument('--input_video', type=str, default=str(Path(gdl.__file__).parents[1] / "data/EMOCA_test_example_data/videos/bilibili.mp4"), 
        help="Filename of the video for reconstruction.")
    parser.add_argument('--output_folder', type=str, default="video_output", help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str, default='EMOCA_v2_lr_mse_20', help='Name of the model to use. Currently EMOCA or DECA are available.')
    parser.add_argument('--path_to_models', type=str, default=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models"))
    parser.add_argument('--mode', type=str, default="detail", choices=["detail", "coarse"], help="Which model to use for the reconstruction.")
    parser.add_argument('--save_images', type=str2bool, default=False, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=str2bool, default=False, help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=str2bool, default=True, help="If true, output meshes will be saved")
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
    parser.add_argument('--include_original', type=str2bool, default=True, 
        help="Apart from the reconstruction video, also a video with the transparent mesh will be added")
    parser.add_argument('--black_background', type=str2bool, default=False, help="If true, the background of the reconstruction video will be black")
    parser.add_argument('--use_mask', type=str2bool, default=True, help="If true, the background of the reconstruction video will be black")
    parser.add_argument('--logger', type=str, default="", choices=["", "wandb"], help="Specify how to log the results if at all.")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    profiler = cProfile.Profile()
    profiler.enable()

    reconstruct_video(args)

    profiler.disable()
    # 输出性能报告
    with open("profile_report_queue.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats("cumtime")  # 可以根据其他统计方式排序，如 'time', 'ncalls'
        stats.print_stats()
        
    print("Done")


if __name__ == '__main__':
    main()
