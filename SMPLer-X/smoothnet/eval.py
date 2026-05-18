import os
import torch



from torch.utils.data import DataLoader

import glob
import numpy as np
import copy
import pickle
from smoothnet_filter import SmoothNetFilter
from scipy.ndimage import gaussian_filter1d



keys = ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'betas', 'expression', 'transl']


    
def save_amass(pose_dict,output_path):
    dict_amass = np.load('ankles_stageii.npz',allow_pickle = True)
    new_dict = {}
    for file in dict_amass.files:
        new_dict[file] = dict_amass[file]
        print('amass',file,dict_amass[file].shape[:])
 
    trans_amass = pose_dict['transl'].reshape(-1, 3)
    root_orient_amass = pose_dict['global_orient'].reshape(-1, 3)
    pose_body_amass = pose_dict['body_pose'].reshape(-1, 63)  
    pose_lhand_amass = pose_dict['left_hand_pose'].reshape(-1, 45)  
    pose_rhand_amass = pose_dict['right_hand_pose'].reshape(-1, 45) 
    pose_hand_amass = np.concatenate([pose_lhand_amass, pose_rhand_amass], axis=-1) 
    pose_jaw_amass = pose_dict['jaw_pose'].reshape(-1, 3)

    betas_amass = np.zeros(16)
    betas_amass[:10] = pose_dict['betas'][0].reshape(-1) 

    new_dict['trans'] = trans_amass
    new_dict['root_orient'] = root_orient_amass
    new_dict['pose_body'] = pose_body_amass
    new_dict['pose_hand'] = pose_hand_amass
    new_dict['pose_jaw'] = pose_jaw_amass
    new_dict['betas'] = betas_amass
    new_dict['markers_latent'][:,:] = 0
    new_dict['pose_eye'] = np.zeros((len(pose_jaw_amass),6))
    poses = np.concatenate([root_orient_amass, pose_body_amass,pose_jaw_amass,new_dict['pose_eye'],pose_hand_amass], axis=-1)
    new_dict['poses'] = poses
    for key in new_dict.keys():
        print('new dict',key,new_dict[key].shape[:])

    np.savez(output_path+'_amass.npz',**new_dict)
    print(f"Poses saved to {output_path}")
    
def restore_and_save_poses(poses, output_path):
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
        'expression': expression
    }
    for key in pose_dict.keys():
        print(key,pose_dict[key].shape[:])

    save_amass(pose_dict,output_path)
    np.savez(output_path+'_smplerx.npz', **pose_dict)
    print(f"Poses saved to {output_path}")


def smooth_sequence_gaussian(sequence, sigma=1):
    smoothed_sequence = np.copy(sequence)
    
    for i in range(3):
        smoothed_sequence[:, i] = gaussian_filter1d(sequence[:, i], sigma=sigma)
    
    return smoothed_sequence

def smooth(all_frames_data_one,output_path, model):
    window_size=8
    step=8
    
    poses = []
    files = []
    for frame_data in all_frames_data_one:
        print(frame_data.keys())
        smplx_data = frame_data['smplx_data']
        pose = []
        for key in ['transl','global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose','betas', 'expression']:
            pose.append(smplx_data[key].reshape(-1))
        pose = np.concatenate(pose)
        poses.append(pose)
    poses = np.array(poses)
    poses_org = copy.deepcopy(poses)
    poses = poses[:,:-20]
    poses_TKC = poses.reshape((len(poses),-1,3))
    smoothnet_poses8 = model(poses_TKC, type='rot')  
    poses_org[:,3:-20] = smoothnet_poses8.reshape((len(smoothnet_poses8),-1))[:,3:]
    poses_org[:,:3] = smooth_sequence_gaussian(poses_org[:,:3], sigma=1)
    restore_and_save_poses(poses_org, output_path)

if __name__ == '__main__':
    model = SmoothNetFilter(
            window_size=8,
            output_size=8,
            checkpoint='./smoothnet_windowsize8.pth.tar')
  
    with open('./clip_5_83.pkl', 'rb') as f:
        input_data = pickle.load(f)
    output_name = 'clip_5-82_1016'    
    smooth(input_data,output_name,model)
