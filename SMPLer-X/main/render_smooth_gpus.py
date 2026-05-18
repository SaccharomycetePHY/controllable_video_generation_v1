import numpy as np
import glob
import random
import cv2
import os
import argparse
import torch
import pyrender
import trimesh
import pandas as pd
import json
import pickle
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import time

from tqdm import tqdm
from multiprocessing import Pool

import smplx
import pdb

smpl_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 'body_pose': (-1, 69)}
smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
        'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3), 'expression': (-1, 10)}
smplx_shape_except_expression = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
        'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3)}

def render_multi_frames(imgs, batch_body_model_params, body_model, cameras, device):
    """
    Render multiple frames using batch processing.
    """
    pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    
    # Prepare batch data
    num_frames = len(imgs)
    
    # Convert body model parameters into a single batch tensor
    batch_params = {}
    for key in batch_body_model_params[0].keys():
        tensors = []
        for param in batch_body_model_params:
            if param[key].dim() == 0:  # Handle scalar tensors
                tensors.append(param[key].unsqueeze(0))
            else:
                tensors.append(param[key])
        batch_params[key] = torch.cat(tensors, dim=0).to(device)
    
    # Run model forward pass in batch mode
    output = body_model(**batch_params, return_verts=True)
    
    vertices_batch = output['vertices'].detach().cpu().numpy()
    faces = body_model.faces
    
    rendered_imgs = []

    # render material
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1,
            roughnessFactor=0.4,
            alphaMode='OPAQUE',
            emissiveFactor=(0.2, 0.2, 0.2),
            baseColorFactor=(0.7, 0.7, 0.7, 1))  
    
    for i in range(num_frames):
        vertices = vertices_batch[i].squeeze()
        body_trimesh = trimesh.Trimesh(vertices, faces, process=False)
        body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        cam_pose = pyrender2opencv @ np.eye(4)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                        ambient_light=(0.3, 0.3, 0.3))
        scene.add(cameras[i], pose=cam_pose)
        scene.add(light, pose=cam_pose)
        scene.add(body_mesh, 'mesh')
        
        img = imgs[i]
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        r = pyrender.OffscreenRenderer(viewport_width=img.shape[1],
                                        viewport_height=img.shape[0],
                                        point_size=1.0)
        
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        img = img / 255
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        output_img = (color[:, :, :] * valid_mask + (~valid_mask) * img)

        img = (output_img * 255).astype(np.uint8)
        rendered_imgs.append(img)
    
    return rendered_imgs

def process_video_chunk(rank, video_paths, args, result_queue):
    # Set device for this process
    device = torch.device(f'cuda:{rank}')
    
    kwargs = dict(gender='neutral',
        num_betas=10,
        use_face_contour=True,
        flat_hand_mean=args.flat_hand_mean,
        use_pca=False,
        batch_size=1)

    smplx_model = smplx.create(
        '../common/utils/human_model_files', 'smplx', 
        **kwargs).to(device)

    for video_path in tqdm(video_paths, desc=f'Processing videos on GPU {rank}'):
        if 'render' in str(video_path):
            continue
            
        cap = cv2.VideoCapture(video_path)
        smplx_path = os.path.join(os.path.dirname(video_path),"smooth_smplx.pkl")
        render_path = os.path.join(os.path.dirname(video_path), 'smpl.mp4')
        
        if os.path.exists(render_path) or not os.path.exists(smplx_path):
            continue

        print(f'Process {rank} rendering: {video_path}')
        
        with open(smplx_path,'rb') as f:
            smplx_data = pickle.load(f)

        out = None
        frame_id = 0
        
        batch_images = []
        batch_body_model_params = []
        batch_cameras = []
        
        while True:
            ret, original_img = cap.read()
            if not ret:
                break
                
            original_img[:,:,:] = 0
            
            body_model_param_tensor = dict()
            for key in smplx_data[frame_id][0].keys():
                body_model_param_tensor[key] = torch.tensor(np.array(smplx_data[frame_id][0][key]), device=device, dtype=torch.float32)
            body_model_param_tensor['right_hand_pose'] = body_model_param_tensor['right_hand_pose'].reshape((1, 15, 3))
            body_model_param_tensor['left_hand_pose'] = body_model_param_tensor['left_hand_pose'].reshape((1, 15, 3))
            body_model_param_tensor['body_pose'] = body_model_param_tensor['body_pose'].reshape((1, 21, 3))

            focal_length = smplx_data[frame_id][0]['focal']
            principal_point = smplx_data[frame_id][0]['princpt']
            camera = pyrender.camera.IntrinsicsCamera(
                    fx=focal_length[0], fy=focal_length[1],
                    cx=principal_point[0], cy=principal_point[1])

            batch_images.append(original_img)
            batch_body_model_params.append(body_model_param_tensor)
            batch_cameras.append(camera)
            
            if len(batch_images) >= args.batch_size:
                rendered_images = render_multi_frames(batch_images, batch_body_model_params, smplx_model, batch_cameras, device)
                
                if out is None:
                    height, width, _ = rendered_images[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(render_path, fourcc, 30, (width, height))
                
                for rendered_image in rendered_images:
                    out.write(rendered_image)
                
                batch_images = []
                batch_body_model_params = []
                batch_cameras = []
            
            frame_id += 1

        # Process remaining frames
        if len(batch_images) > 0:
            rendered_images = render_multi_frames(batch_images, batch_body_model_params, smplx_model, batch_cameras, device)
            
            if out is None:
                height, width, _ = batch_images[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(render_path, fourcc, 30, (width, height))
            
            for rendered_image in rendered_images:
                out.write(rendered_image)

        if out is not None:
            out.release()
            print(f'Process {rank} saved: {render_path}')
            
        cap.release()

def visualize_seqs(args):
    root_folder = args.root_path
    print('Starting multi-GPU rendering')

    video_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('clip.mp4') and os.path.exists(os.path.join(dirpath, 'smooth_smplx.pkl')):
                if os.path.exists(os.path.join(dirpath, 'smpl.mp4')):
                    continue
                video_paths.append(os.path.join(dirpath, filename))
    print('共处理',len(video_paths),'个视频')

    num_gpus = torch.cuda.device_count()
    print(f'Found {num_gpus} GPUs')

    chunks = np.array_split(video_paths, num_gpus)
    
    mp.set_start_method('spawn', force=True)
    processes = []
    result_queue = Queue()

    for rank in range(num_gpus):
        p = Process(target=process_video_chunk, 
                   args=(rank, chunks[rank], args, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=False,
                        help='path to the data folder')
    parser.add_argument('--load_mode', type=str, required=False,
                        default='smplerx',
                        help='load mode: smplerx or other test mode, please select smplerx')
    parser.add_argument('--seq', type=str, required=False,
                        help='seq name or seq pattern',
                        default='default')
    parser.add_argument('--image_path', type=str, required=False,
                        help='path to the image folder')
    parser.add_argument('--root_path', type=str, required=False, default='/amax/zyude/human_video_data_all/results_11_14',
                        help='path to the image folder')
    parser.add_argument('--flat_hand_mean', type=bool, required=False,
                        help='use flat hand mean for smplx',
                        default=False)
    parser.add_argument('--render_biggest_person', type=str, required=False,
                        help='render biggest person in the frame',
                        default='True')
    parser.add_argument('--batch_size', type=int, required=False,
                        default=2048,
                        help='batch size for rendering')

    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    start_time = time.time()
    visualize_seqs(args)
    end_time = time.time()
    print(f'Total time: {end_time - start_time:.2f}s')