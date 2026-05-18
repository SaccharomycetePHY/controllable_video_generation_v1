import numpy as np
import glob
import random
import cv2
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import os
import argparse
import torch
import pyrender
import trimesh
import json
import pickle

from tqdm import tqdm
from multiprocessing import Pool

# from mmhuman3d.models.body_models.builder import build_body_model
import smplx
import pdb

smpl_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 'body_pose': (-1, 69)}
smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
        'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3), 'expression': (-1, 10)}
smplx_shape_except_expression = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
        'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3)}
# smplx_shape = smplx_shape_except_expression

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



def visualize_seqs(args):

    kwargs = dict(gender='neutral',
        num_betas=10,
        use_face_contour=True,
        flat_hand_mean=args.flat_hand_mean,
        use_pca=False,
        batch_size=1)

    smplx_model = smplx.create(
        '../common/utils/human_model_files', 'smplx', 
        **kwargs).to(args.device)
    

    root_folder = "/home/lm/Datahouse/En3d/tmp_test"
    print('start')
    for dirpath, _, filenames in os.walk(root_folder):
        # print(dirpath)
        for filename in filenames:
            
            # exit()
            if filename.endswith('clip.mp4'):
                # import pdb
                # pdb.set_trace()
            
                video_path = os.path.join(dirpath, filename)
                print(video_path)
                if 'render' in str(video_path):
                    continue
                

                cap = cv2.VideoCapture(video_path)
                smplx_path = os.path.join(os.path.dirname(video_path), "smooth_smplx","clip.pkl")
                render_path = os.path.join(os.path.dirname(video_path), 'smooth_render.mp4')
                print('smplx_path',smplx_path)
                if not os.path.exists(smplx_path):
                    print('not exists')
                    continue
                with open(smplx_path,'rb') as f:

                    smplx_data = pickle.load(f)
                    print('load pkl')

                out_mp4 = render_path
                out = None

                frame_id = 0
                while True:
                    ret, original_img = cap.read()  
                    # print(frame_id)
                    
                    if not ret:
                        break
                    original_img[:,:,:] = 0

                        

                    body_model_params = []
                    cameras = []
                    image = original_img

                    for itm in smplx_data[frame_id]:
                        focal_length = itm['focal']
                        principal_point = itm['princpt']


                        camera = pyrender.camera.IntrinsicsCamera(
                                fx=focal_length[0], fy=focal_length[1],
                                cx=principal_point[0], cy=principal_point[1],)



                        body_model_param_tensor = dict()
                        for key in itm.keys():
                            body_model_param_tensor[key] = torch.tensor(np.array(itm[key]), device=args.device, dtype=torch.float32)
                        body_model_param_tensor['right_hand_pose'] = body_model_param_tensor['right_hand_pose'].reshape((1, 15, 3))
                        body_model_param_tensor['left_hand_pose'] = body_model_param_tensor['left_hand_pose'].reshape((1, 15, 3))
                        body_model_param_tensor['body_pose'] = body_model_param_tensor['body_pose'].reshape((1, 21, 3))
                        

                        cameras.append(camera)
                        body_model_params.append(body_model_param_tensor)
                        # bbox_sizes.append(bbox_size)

                    if args.render_biggest_person == 'True':
                        # bid = bbox_sizes.index(max(bbox_sizes))
                        bid = 0
                        rendered_image = render_pose(img=image,
                                        body_model_param=body_model_params[bid],
                                        body_model=smplx_model,
                                        camera=cameras[bid])
                    # 合并原图和渲染图
                    # combined_image = np.hstack((image, rendered_image))
                    combined_image = rendered_image
                
                    # save_name = os.path.join(save_path, f'{int(framestamp):06d}.jpg')
                    if out is None:
                        # 获取第一帧的宽度和高度
                        height, width, _ = combined_image.shape

                        # 根据图像大小和帧率创建 VideoWriter
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                       
                        out = cv2.VideoWriter(out_mp4, fourcc, 30, (width, height))
                        print('define',out_mp4)
                    
                    # 写入帧
                    out.write(combined_image)
                    frame_id += 1
                    # cv2.imwrite('test.jpg', combined_image)
                if out is not None:
                    out.release()
                    print('save at',out_mp4)
               
           
            

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

    # optional args
    parser.add_argument('--flat_hand_mean', type=bool, required=False,
                        help='use flat hand mean for smplx',
                        default=False)
    parser.add_argument('--render_biggest_person', type=str, required=False,
                        help='render biggest person in the frame',
                        default='True')
    args = parser.parse_args()

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # import pdb
    # pdb.set_trace()
    print(args.device)
    visualize_seqs(args)