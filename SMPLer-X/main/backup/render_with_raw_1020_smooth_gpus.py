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

def render_pose(img, body_model_param, body_model, camera, return_mask=False):

    # the inverse is same
    pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    
    output = body_model(**body_model_param, return_verts=True)
    
    vertices = output['vertices'].detach().cpu().numpy().squeeze()
    faces = body_model.faces

    # render material
    base_color = (1.0, 193/255, 193/255, 1.0)
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0,
            alphaMode='OPAQUE',
            baseColorFactor=base_color)
    
    material_new = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1,
            roughnessFactor=0.4,
            alphaMode='OPAQUE',
            emissiveFactor=(0.2, 0.2, 0.2),
            baseColorFactor=(0.7, 0.7, 0.7, 1))  
    material = material_new
    
    # get body mesh
    body_trimesh = trimesh.Trimesh(vertices, faces, process=False)
    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

    # prepare camera and light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    cam_pose = pyrender2opencv @ np.eye(4)
    
    # build scene
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                    ambient_light=(0.3, 0.3, 0.3))
    scene.add(camera, pose=cam_pose)
    scene.add(light, pose=cam_pose)
    scene.add(body_mesh, 'mesh')

    # render scene
    os.environ["PYOPENGL_PLATFORM"] = "osmesa" # include this line if use in vscode
    r = pyrender.OffscreenRenderer(viewport_width=img.shape[1],
                                    viewport_height=img.shape[0],
                                    point_size=1.0)
    
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    # alpha = 1.0  # set transparency in [0.0, 1.0]
    # color[:, :, -1] = color[:, :, -1] * alpha
    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    img = img / 255
    # output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * img)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    output_img = (color[:, :, :] * valid_mask + (~valid_mask) * img)

    # output_img = color

    img = (output_img * 255).astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if return_mask:
        return img, valid_mask, (color * 255).astype(np.uint8)

    return img


def render_multi_pose(img,
                      body_model_params,
                      body_model,
                      cameras):

    masks, colors = [], []

    # calculate distance based on transl
    dists, valid_idx = [], []
    for i, body_model_param in enumerate(body_model_params):
        dist = np.linalg.norm(body_model_param['transl'].detach().cpu()) * 2/ (cameras[i].fx + cameras[i].fy)
        if dist not in dists:
            valid_idx.append(i)
            dists.append(dist)

    # pdb.set_trace()

    # select by valid idx
    body_model_params = [body_model_params[i] for i in valid_idx]
    cameras = [cameras[i] for i in valid_idx]

    # sort by dist

    body_model_params = [x for _, x in sorted(zip(dists, body_model_params), reverse=True)]
    cameras = [x for _, x in sorted(zip(dists, cameras), reverse=True)]


    # render separate masks
    for i, body_model_param in enumerate(body_model_params):

        _, mask, color = render_pose(
            img=img,
            body_model_param=body_model_param,
            body_model=body_model,
            camera=cameras[i],
            return_mask=True,
        )
        masks.append(mask)
        colors.append(color)
    # sum masks
    mask_sum = np.sum(masks, axis=0)
    mask_all = (mask_sum > 0)

    # pp_occ = 1 - np.sum(mask_all) / np.sum(mask_sum)

    # overlay colors to img
    for i, color in enumerate(colors):
        mask = masks[i]
        img = img * (~mask) + color * mask

    img = img.astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def render_frame(framestamp, anno_ps, image_base_path, seq, smplx_model, args):
    annos = [p for p in anno_ps if framestamp in os.path.basename(p)]
    annos = [p for p in annos if 'person' not in os.path.basename(p)]

    body_model_params = []
    cameras = []
    bbox_sizes = []
    try:
        # image_path = os.path.join(seq, f'0{framestamp}.jpg').replace(args.data_path, args.image_path)
        image_path = os.path.join(image_base_path, f'0{framestamp}.jpg')
        # pdb.set_trace()
        image = cv2.imread(image_path)
    except:

        pass
    # pdb.set_trace()
    for anno_p in annos:

        anno = dict(np.load(anno_p, allow_pickle=True))

        meta = json.load(open(os.path.join(seq, 'meta', 
                                        os.path.basename(anno_p).replace('.npz', '.json')
                                        )))

        # bbox_size = meta['bbox'][2] * meta['bbox'][3]
        focal_length = meta['focal']
        principal_point = meta['princpt']
        camera = pyrender.camera.IntrinsicsCamera(
                fx=focal_length[0], fy=focal_length[1],
                cx=principal_point[0], cy=principal_point[1],)

        # prepare body model params
        intersect_key = list(set(anno.keys()) & set(smplx_shape.keys()))
        body_model_param_tensor = {key: torch.tensor(
                np.array(anno[key]).reshape(smplx_shape[key]), device=args.device, dtype=torch.float32)
                        for key in intersect_key if len(anno[key]) > 0}
        
        cameras.append(camera)
        body_model_params.append(body_model_param_tensor)
        bbox_sizes.append(bbox_size)

    # render pose
    if args.render_biggest_person == 'True':
        bid = bbox_sizes.index(max(bbox_sizes))
        rendered_image = render_pose(img=image,
                        body_model_param=body_model_params[bid],
                        body_model=smplx_model,
                        camera=cameras[bid])
    else:
        rendered_image = render_multi_pose(img=image, 
                        body_model_params=body_model_params, 
                        body_model=smplx_model,
                        cameras=cameras)

    sp = seq.replace(f'{args.data_path}{os.path.sep}', '')
    save_path = os.path.join(args.data_path, 'output', sp)
    os.makedirs(save_path, exist_ok=True)

    save_name = os.path.join(save_path, framestamp+'.jpg')
    cv2.imwrite(save_name, rendered_image)


def call_frame_render(args):
    return render_frame(*args)

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
        
        while True:
            ret, original_img = cap.read()
            if not ret:
                break
                
            original_img[:,:,:] = 0
            
            body_model_params = []
            cameras = []
            
            for itm in smplx_data[frame_id]:
                focal_length = itm['focal']
                principal_point = itm['princpt']

                camera = pyrender.camera.IntrinsicsCamera(
                        fx=focal_length[0], fy=focal_length[1],
                        cx=principal_point[0], cy=principal_point[1])

                body_model_param_tensor = dict()
                for key in itm.keys():
                    body_model_param_tensor[key] = torch.tensor(np.array(itm[key]), device=device, dtype=torch.float32)
                body_model_param_tensor['right_hand_pose'] = body_model_param_tensor['right_hand_pose'].reshape((1, 15, 3))
                body_model_param_tensor['left_hand_pose'] = body_model_param_tensor['left_hand_pose'].reshape((1, 15, 3))
                body_model_param_tensor['body_pose'] = body_model_param_tensor['body_pose'].reshape((1, 21, 3))

                cameras.append(camera)
                body_model_params.append(body_model_param_tensor)

            if args.render_biggest_person == 'True':
                rendered_image = render_pose(img=original_img,
                                body_model_param=body_model_params[0],
                                body_model=smplx_model,
                                camera=cameras[0])
            else:
                rendered_image = render_multi_pose(img=original_img, 
                                body_model_params=body_model_params, 
                                body_model=smplx_model,
                                cameras=cameras)

            if out is None:
                height, width, _ = rendered_image.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(render_path, fourcc, 30, (width, height))

            out.write(rendered_image)
            frame_id += 1

        if out is not None:
            out.release()
            print(f'Process {rank} saved: {render_path}')
            
        cap.release()

def visualize_seqs(args):
    root_folder = '/amax/zyude/human_video_data_all/data_one_person'
    print('Starting multi-GPU rendering')

    # Get all video paths
    video_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('clip.mp4') and os.path.exists(os.path.join(dirpath, 'smooth_smplx.pkl')):
                if os.path.exists(os.path.join(dirpath, 'smpl.mp4')):
                    continue
                video_paths.append(os.path.join(dirpath, filename))
    print('共处理',len(video_paths),'个视频')

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f'Found {num_gpus} GPUs')

    # Split videos among GPUs
    chunks = np.array_split(video_paths, num_gpus)
    
    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)
    processes = []
    result_queue = Queue()

    # Start processes
    for rank in range(num_gpus):
        p = Process(target=process_video_chunk, 
                   args=(rank, chunks[rank], args, result_queue))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
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
    parser.add_argument('--root_path', type=str, required=False,
                        help='path to the image folder')
    parser.add_argument('--flat_hand_mean', type=bool, required=False,
                        help='use flat hand mean for smplx',
                        default=False)
    parser.add_argument('--render_biggest_person', type=str, required=False,
                        help='render biggest person in the frame',
                        default='True')

    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    visualize_seqs(args)