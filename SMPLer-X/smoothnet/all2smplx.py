import numpy as np
import os
import glob

def load_smplx_parameters(npz_path):
    return np.load(npz_path)

def save_single_frame_npz(data, idx, output_dir):
    frame_data = {
        'global_orient': data['global_orient'][idx],
        'body_pose': data['body_pose'][idx],
        'betas': data['betas'][idx],
        'transl': data['transl'][idx],
        'left_hand_pose': data['left_hand_pose'][idx],
        'right_hand_pose': data['right_hand_pose'][idx],
        'jaw_pose': data['jaw_pose'][idx],
        'leye_pose': data['leye_pose'][idx],
        'reye_pose': data['reye_pose'][idx],
        'expression': data['expression'][idx]
    }
    
    existing_files = glob.glob(os.path.join(output_dir, f'{idx+1:05d}_*.npz'))
    if not existing_files:
        existing_files = [os.path.join(output_dir, f'{idx+1:05d}_0.npz')]
    
    for file in existing_files:
        np.savez(file, **frame_data)
        print(f'已覆盖文件: {file}')

# npz_path = '/amax/zyude/controllable_video_generation/SMPLer-X/smoothnet/clip_5-82_3_smplerx.npz'
# output_dir = '/amax/zyude/controllable_video_generation/SMPLer-X/clips/results/clip_5-82_3/smplx'
npz_path = '/amax/zyude/controllable_video_generation/SMPLer-X/smoothnet/clip_5-82_3_smplerx.npz'
output_dir = '/amax/zyude/controllable_video_generation/SMPLer-X/clips/results/clip_5-82_3/smplx'


os.makedirs(output_dir, exist_ok=True)

data = load_smplx_parameters(npz_path)
num_frames = len(data['transl'])

for idx in range(num_frames):
    save_single_frame_npz(data, idx, output_dir)

print(f'总共处理了 {num_frames} 帧数据')
