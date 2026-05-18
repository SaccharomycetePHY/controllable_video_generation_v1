import os
import torch



from torch.utils.data import DataLoader

import glob
import numpy as np
import copy
import pickle
from smoothnet_filter import SmoothNetFilter
from scipy.ndimage import gaussian_filter1d

import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

# from vis_points import optimize_points
import subprocess



keys = ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'betas', 'expression', 'transl']

    
def restore_and_save_poses(poses, smplx_data_list,save_path):
    # 先确定各个 key 的尺寸
    transl_size = 3
    global_orient_size = 3
    body_pose_size = 21 * 3
    left_hand_pose_size = 15 * 3
    right_hand_pose_size = 15 * 3
    jaw_pose_size = 3
    leye_pose_size = 3
    reye_pose_size = 3
    betas_size = 10
    expression_size = 10


    transl_end = transl_size
    global_orient_end = transl_end + global_orient_size
    body_pose_end = global_orient_end + body_pose_size
    left_hand_pose_end = body_pose_end + left_hand_pose_size
    right_hand_pose_end = left_hand_pose_end + right_hand_pose_size
    jaw_pose_end = right_hand_pose_end + jaw_pose_size
    leye_pose_end = jaw_pose_end + leye_pose_size
    reye_pose_end = leye_pose_end + reye_pose_size
    betas_end = reye_pose_end + betas_size
    expression_end = betas_end + expression_size

    transl = poses[:, :transl_end].reshape(-1, 1, 3)
    global_orient = poses[:, transl_end:global_orient_end].reshape(-1, 1, 3)
    body_pose = poses[:, global_orient_end:body_pose_end].reshape(-1, 21, 3)
    left_hand_pose = poses[:, body_pose_end:left_hand_pose_end].reshape(-1, 15, 3)
    right_hand_pose = poses[:, left_hand_pose_end:right_hand_pose_end].reshape(-1, 15, 3)
    jaw_pose = poses[:, right_hand_pose_end:jaw_pose_end].reshape(-1, 1, 3)
    leye_pose = poses[:, jaw_pose_end:leye_pose_end].reshape(-1, 1, 3)
    reye_pose = poses[:, leye_pose_end:reye_pose_end].reshape(-1, 1, 3)
    betas = poses[:, reye_pose_end:betas_end].reshape(-1, 1, 10)
  
    expression = poses[:, betas_end:expression_end].reshape(-1, 1, 10)
    focal = poses[:, -4:-2].reshape(-1, 1, 2)
    princpt = poses[:, -2:].reshape(-1, 1, 2)
    

    
    # 将数据保存为字典
    pose_dict = {
        'transl': transl,
        'global_orient': global_orient,
        'body_pose': body_pose,
        'left_hand_pose': left_hand_pose,
        'right_hand_pose': right_hand_pose,
        'jaw_pose': jaw_pose,
        'leye_pose': leye_pose,
        'reye_pose': reye_pose,
        'betas': betas,
        'expression': expression,
        'focal':focal,
        'princpt':princpt,
        
    }
    frame_id = 0
    for itm in smplx_data_list:
        # print(itm)
        for key in pose_dict.keys():
            itm[0][key] = np.array(pose_dict[key][frame_id]).reshape( np.array(itm[0][key]).shape)
        frame_id += 1
    with open(save_path, 'wb') as f:
        pickle.dump(smplx_data_list, f)
            

    
    # np.savez(output_path+'_smplerx.npz', **pose_dict)
    print(f"Poses saved to {save_path}")
 
    


def smooth_sequence_gaussian(sequence, sigma=1):
    smoothed_sequence = np.copy(sequence)
    
    for i in range(sequence.shape[1]):
        smoothed_sequence[:, i] = gaussian_filter1d(sequence[:, i], sigma=sigma)
    
    return smoothed_sequence

def smooth(smplx_data_list,save_path, model):
    window_size=8
    step=8
    
    poses = []
    files = []
    for frame_data in smplx_data_list:
        if len(frame_data) <1:
            print(f"Frame data has less than 1 item")
            break
        
        smplx_data = frame_data[0]#0号人
        pose = []
        for key in ['transl','global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose','betas', 'expression','focal','princpt']:
            # print(key,smplx_data[key])
            pose.append(np.array(smplx_data[key]).reshape(-1))
        pose = np.concatenate(pose)
        poses.append(pose)
    poses = np.array(poses)
    poses_org = copy.deepcopy(poses)

    if poses.shape==(0,):
        with open('/amax/zyude/miss.txt', 'a') as file:  # 
            file.write(f'{save_path} have 0 frames \n')
        return
    poses = poses[:,:-24]
    poses_TKC = poses.reshape((len(poses),-1,3))
    smoothnet_poses8 = model(poses_TKC, type='rot')  
    poses_org[:,3:-24] = smoothnet_poses8.reshape((len(smoothnet_poses8),-1))[:,3:]
    poses_org[:,:3] = smooth_sequence_gaussian(poses_org[:,:3], sigma=1)
    poses_org[:,-4:] = smooth_sequence_gaussian(poses_org[:,-4:], sigma=1)
    print(poses_org.shape[:])#(80, 192)
    if poses_org.shape[:][0]!=120:
        return
    restore_and_save_poses(poses_org, smplx_data_list,save_path)

if __name__ == '__main__':
    model = SmoothNetFilter(
            window_size=8,
            output_size=8,
            checkpoint='./smoothnet_windowsize8.pth.tar')

    # root_folder = '/amax/zejian/DATA/En3d/'
    # root_folder = '/home/lm/Datahouse/En3d/dataset2'
    # root_folder = '/amax/zyude/data_mimo'
    root_folder = '/home/lm/Datahouse/En3d/dataset_En3d_fixcam'
    
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('clip.mp4'):
                # import pdb
                # pdb.set_trace()
            
                video_path = os.path.join(dirpath, filename)
                print(video_path)
                if 'render' in str(video_path):
                    continue
               
                # smplx_dir = os.path.join(os.path.dirname(video_path), 'smplx')
                smplx_dir=os.path.dirname(video_path)
                
                if not os.path.exists(smplx_dir):
                    
                    break

                save_path = os.path.join(smplx_dir, 'smooth_smplx.pkl')

                if os.path.exists(save_path):
                    print(save_path+" exist")
                    break
                file_path = os.path.join(smplx_dir, 'smplx', 'clip.pkl')
                print('file_path',file_path)
                if not os.path.exists(file_path):
                    break
                with open(file_path,'rb') as f:

                    smplx_data = pickle.load(f)

                smooth(smplx_data,save_path,model)

                if os.path.exists(os.path.join(smplx_dir,"smooth_smplx")):
                    subprocess.run(["rm","-rf",os.path.join(smplx_dir,"smooth_smplx")])

                print(save_path,"OK")

              

  
    # with open('../SmoothNet-main/find_emoca_ids/clip_5_83_with_cam.pkl', 'rb') as f:
    #     input_data = pickle.load(f)
    # output_name = 'clip_5-82_smooth_cam_root_together_-1_1017'    
    # smooth(input_data,output_name,model)
