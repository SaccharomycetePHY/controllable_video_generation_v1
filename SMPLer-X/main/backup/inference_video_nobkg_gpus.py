import os
import torch.multiprocessing as mp
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
import cv2
from tqdm import tqdm
import json
from typing import Literal, Union
from mmdet.apis import init_detector, inference_detector
from utils.inference_utils import process_mmdet_results, non_max_suppression
import pickle
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=str, dest='num_gpus',default = '1')
    parser.add_argument('--exp_name', type=str, default='output/demo_inference_clip_5-82_')
    parser.add_argument('--pretrained_model', type=str, default='smpler_x_h32')
    parser.add_argument('--testset', type=str, default='EHF')
    parser.add_argument('--agora_benchmark', type=str, default='agora_model')
    parser.add_argument('--img_path', type=str, default='input.png')
    parser.add_argument('--start', type=str, default=1)
    parser.add_argument('--end', type=str, default=1)
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--demo_dataset', type=str, default='na')
    parser.add_argument('--demo_scene', type=str, default='all')
    parser.add_argument('--show_verts', action="store_true")
    parser.add_argument('--show_bbox', action="store_true")
    parser.add_argument('--save_mesh', action="store_true")
    parser.add_argument('--multi_person', action="store_true")
    parser.add_argument('--iou_thr', type=float, default=0.2)
    parser.add_argument('--bbox_thr', type=int, default=20)
    parser.add_argument('--root_path', type=str, default='/amax/zyude/human_video_data_all/data_one_person')
    # parser.add_argument('--video_path', type=str)
    parser.add_argument('--gpu', type=str,default = '1')
    parser.add_argument('--cut', type=int,default = 1)
  
    args = parser.parse_args()
    return args
    #  CUDA_VISIBLE_DEVICES=7 python inference.py --num_gpus 1 --exp_name              
    #        output/demo_inference_clip_5-82_ --pretrained_model smpler_x_h32
    # --agora_benchmark ago                     ra_model --img_path ../clips/images/clip_5-82_ 
    #--start 1 --end 78 --output_folder ../c                     lips/results/clip_5-82_ 
    #--show_verts --show_bbox --save_mesh --multi_person --iou_thr                     
    # 0.2 --bbox_thr 20


def process_videos(gpu_id, gpu_videos, num_gpus, args):
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)

    config_path = osp.join('./config', f'config_{args.pretrained_model}.py')
    ckpt_path = osp.join('../pretrained_models', f'{args.pretrained_model}.pth.tar')

    cfg.get_config_fromfile(config_path)
    cfg.update_test_config(args.testset, args.agora_benchmark, shapy_eval_split=None, 
                            pretrained_model_path=ckpt_path, use_cache=False)
    cfg.update_config(args.num_gpus, args.exp_name)
    cudnn.benchmark = True

    # load model
    from base import Demoer
    from utils.preprocessing import load_img, process_bbox, generate_patch_image
    from utils.vis import render_mesh, save_obj
    from utils.human_models import smpl_x
    demoer = Demoer()
    demoer._make_model()
    demoer.model = demoer.model.to(device)
    demoer.model.eval()
    
    start = int(args.start)
    end = start + int(args.end)
    multi_person = args.multi_person
            

    ### mmdet init
    checkpoint_file = '../pretrained_models/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    config_file= '../pretrained_models/mmdet/mmdet_faster_rcnn_r50_fpn_coco.py'
    model = init_detector(config_file, checkpoint_file, device=device)
    
    try:
        for video_dir, clip_name, clip_dir in tqdm(gpu_videos, desc=f"处理GPU {device}的视频"):
            filename = "clip.mp4"
            video_path = os.path.join(clip_dir, filename)
            print(f"正在处理视频: {video_path}")
            if 'render' in str(video_path):
                continue
            

            cap = cv2.VideoCapture(video_path)
            save_path = os.path.join(os.path.dirname(video_path), 'smplx')

            
            os.makedirs(save_path, exist_ok=True)
            basename = os.path.basename(video_path).split('.')[0]                                                                                                                                                                                                                                                                                                                   
            all_frams = []
            file_path = os.path.join(save_path, basename + '.pkl')
            
            start_time = time.time()

            while True:
                ret, original_img = cap.read()  
                if not ret:
                    with open(os.path.join(save_path,basename+'.pkl'), 'wb') as f:
                        
                        pickle.dump(all_frams, f)
                    print('all_frams',len(all_frams))
                    print('average time',(time.time()-start_time)/len(all_frams))
                    break

                # prepare input image
                transform = transforms.ToTensor()
            
                vis_img = original_img.copy()
                original_img_height, original_img_width = original_img.shape[:2]
                os.makedirs(args.output_folder, exist_ok=True)

                ## mmdet inference
                mmdet_results = inference_detector(model, original_img)
                mmdet_box = process_mmdet_results(mmdet_results, cat_id=0, multi_person=True)
                
                # save original image if no bbox
                if len(mmdet_box[0])<1:
                    # with open('miss.txt', 'a') as file:  # 
                    #     file.write(f'{video_path}\n')
                    print('do not have person')
                    break#整个mp4不要了
                
                if not multi_person:
                    # only select the largest bbox
                    num_bbox = 1
                    mmdet_box = mmdet_box[0]
                else:
                    # keep bbox by NMS with iou_thr
                    mmdet_box = non_max_suppression(mmdet_box[0], args.iou_thr)
                    num_bbox = len(mmdet_box)
                
                ## loop all detected bboxes
                all_person = []
                for bbox_id in range(num_bbox):
                    mmdet_box_xywh = np.zeros((4))
                    mmdet_box_xywh[0] = mmdet_box[bbox_id][0]
                    mmdet_box_xywh[1] = mmdet_box[bbox_id][1]
                    mmdet_box_xywh[2] =  abs(mmdet_box[bbox_id][2]-mmdet_box[bbox_id][0])
                    mmdet_box_xywh[3] =  abs(mmdet_box[bbox_id][3]-mmdet_box[bbox_id][1]) 

                    # skip small bboxes by bbox_thr in pixel
                    if mmdet_box_xywh[2] < args.bbox_thr or mmdet_box_xywh[3] < args.bbox_thr * 3:
                        continue

                    # for bbox visualization 
                    start_point = (int(mmdet_box[bbox_id][0]), int(mmdet_box[bbox_id][1]))
                    end_point = (int(mmdet_box[bbox_id][2]), int(mmdet_box[bbox_id][3]))   

                    bbox = process_bbox(mmdet_box_xywh, original_img_width, original_img_height)
                    img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
                    img = transform(img.astype(np.float32))/255
                    img = img[None,:,:,:].to(device)
                    inputs = {'img': img}
                    targets = {}
                    meta_info = {}

                    # mesh recovery
                    with torch.no_grad():
                        out = demoer.model(inputs, targets, meta_info, 'test')
                    mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

                    ## save mesh
                    if args.save_mesh:
                        save_path_mesh = os.path.join(args.output_folder, 'mesh')
                        os.makedirs(save_path_mesh, exist_ok= True)
                        save_obj(mesh, smpl_x.face, os.path.join(save_path_mesh, f'{frame:05}_{bbox_id}.obj'))

                    ## save single person param
                    smplx_pred = {}
                    smplx_pred['global_orient'] = out['smplx_root_pose'].reshape(-1,3).cpu().numpy()
                    smplx_pred['body_pose'] = out['smplx_body_pose'].reshape(-1,3).cpu().numpy()
                    smplx_pred['left_hand_pose'] = out['smplx_lhand_pose'].reshape(-1,3).cpu().numpy()
                    smplx_pred['right_hand_pose'] = out['smplx_rhand_pose'].reshape(-1,3).cpu().numpy()
                    smplx_pred['jaw_pose'] = out['smplx_jaw_pose'].reshape(-1,3).cpu().numpy()
                    smplx_pred['leye_pose'] = np.zeros((1, 3))
                    smplx_pred['reye_pose'] = np.zeros((1, 3))
                    smplx_pred['betas'] = out['smplx_shape'].reshape(-1,10).cpu().numpy()
                    smplx_pred['expression'] = out['smplx_expr'].reshape(-1,10).cpu().numpy()
                    smplx_pred['transl'] =  out['cam_trans'].reshape(-1,3).cpu().numpy()
                    # save_path_smplx = os.path.join(args.output_folder, 'smplx')
                    # os.makedirs(save_path_smplx, exist_ok= True)

                    

                    ## render single person mesh
                    focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
                    princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
                    ## save single person meta
                    meta = {'focal': focal, 
                            'princpt': princpt, 
                            'bbox': bbox.tolist(), 
                            'bbox_mmdet': mmdet_box_xywh.tolist(), 
                            'bbox_id': bbox_id}
                    # json_object = json.dumps(meta, indent=4)
                    for key in meta.keys():
                        smplx_pred[key]=meta[key]
                    smplx_pred['bbox_id'] = bbox_id

                    all_person.append(smplx_pred)
                all_frams.append(all_person)
    except Exception as e:
        print(f"处理视频时发生错误: {e}")

def main():

    args = parse_args()
    root_dir = args.root_path

    all_videos = []
    for video_name in os.listdir(root_dir):
        video_dir = os.path.join(root_dir, video_name)
        if os.path.isdir(video_dir):
            for clip_name in os.listdir(video_dir):
                clip_base_name = clip_name[:-4] if clip_name.endswith(".mp4") else clip_name
                clip_dir = os.path.join(video_dir, clip_base_name)
                if os.path.exists(os.path.join(clip_dir, "smplx.pkl")):
                    print(f"已存在smplx.pkl文件: {clip_dir}")
                    continue
                all_videos.append((video_dir, clip_name, clip_dir))

    print(f"共有 {len(all_videos)} 个视频需要处理")
    # 多进程处理
    num_gpus = torch.cuda.device_count()
    videos_per_gpu = len(all_videos) // num_gpus
    processes = []

    for gpu_id in range(num_gpus):
        start_idx = gpu_id * videos_per_gpu
        end_idx = start_idx + videos_per_gpu if gpu_id < num_gpus - 1 else len(all_videos)
        gpu_videos = all_videos[start_idx:end_idx]
        
        p = mp.Process(target=process_videos, args=(gpu_id, gpu_videos, num_gpus, args))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()