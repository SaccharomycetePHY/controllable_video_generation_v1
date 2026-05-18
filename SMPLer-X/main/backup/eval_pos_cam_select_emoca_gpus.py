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
import shutil
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
from tqdm import tqdm

def draw_bbox(image, bbox, color, label):
    x, y, w, h = bbox
    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
    cv2.putText(image, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def visualize_video_with_bbox(video_path, emoca_centers_sizes, all_frames_data, all_frames_data_one, output_video_path):
    for i in range(len(all_frames_data_one)):
        all_frames_data_one[i] = all_frames_data_one[i][0]
    cap = cv2.VideoCapture(video_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_index < len(emoca_centers_sizes):
            emoca_bbox = emoca_centers_sizes[frame_index]
            draw_bbox(frame, emoca_bbox, (255, 0, 0), "Face")

        if frame_index < len(all_frames_data_one) and all_frames_data_one[frame_index] is not None:
            matched_person_data = all_frames_data_one[frame_index]
            body_bbox = matched_person_data['bbox_mmdet']
            draw_bbox(frame, body_bbox, (0, 255, 0), "Matched Body")

        if frame_index < len(all_frames_data):
            all_persons = all_frames_data[frame_index]
            for person_data in all_persons:
                body_bbox = person_data['bbox_mmdet']
                if matched_person_data is None or body_bbox != matched_person_data['bbox_mmdet']:
                    draw_bbox(frame, body_bbox, (0, 0, 255), "Unmatched Body")

        out.write(frame)
        frame_index += 1

        if frame_index % 50 == 0:
            print(f"Processed {frame_index}/{frame_count} frames")

    cap.release()
    out.release()
    print(f"保存视频到: {output_video_path}")

def is_bbox_contained(face_bbox, body_bbox):
    face_x, face_y, face_w, face_h = face_bbox
    body_x, body_y, body_w, body_h = body_bbox

    face_center_x = face_x + face_w / 2
    face_center_y = face_y + face_h / 2

    return (body_x <= face_center_x <= body_x + body_w and
            body_y <= face_center_y <= body_y + body_h)

def find_best_match_in_frame(emoca_bbox, frame_data):
    best_id = 0
    idx = 0 
    is_find = False
    for person_data in frame_data:
        smplx_bbox = person_data['bbox_mmdet']
        if is_bbox_contained(emoca_bbox, smplx_bbox):
            best_id = idx
            is_find = True
            break
        idx += 1
    if not is_find:
        print('is find',is_find)
        return -1

    return best_id

def match_emoca_to_smplx(emoca_centers_sizes, all_frames_data,is_emoca_exists = True):
    import copy
   
    all_frames_data_one = copy.deepcopy(all_frames_data)
    prev_best_person_id = None  

    cnt, last_best_id = 0, 0
    for frame_index, frame_data in enumerate(all_frames_data):
        emoca_bbox = emoca_centers_sizes[frame_index]
        best_id = find_best_match_in_frame(emoca_bbox, frame_data)
        if best_id == -1:
            all_frames_data_one[frame_index][0] = all_frames_data_one[frame_index][last_best_id]
            cnt += 1
            continue
        
        all_frames_data_one[frame_index][0] = all_frames_data_one[frame_index][best_id]
        last_best_id = best_id

    if cnt > 0.8 * len(all_frames_data):
        drop_video = True
    else:
        drop_video = False
        
    return all_frames_data_one, drop_video

keys = ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'betas', 'expression', 'transl']
    
def restore_and_save_poses(poses, smplx_data_list,save_path):
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
        for key in pose_dict.keys():
            itm[0][key] = np.array(pose_dict[key][frame_id]).reshape( np.array(itm[0][key]).shape)
        frame_id += 1
    with open(save_path, 'wb') as f:
        pickle.dump(smplx_data_list, f)
            
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
            break
        
        smplx_data = frame_data[0]
        pose = []
        for key in ['transl','global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose','betas', 'expression','focal','princpt']:
            pose.append(np.array(smplx_data[key]).reshape(-1))
        pose = np.concatenate(pose)
        poses.append(pose)
    poses = np.array(poses)
    poses_org = copy.deepcopy(poses)
    poses = poses[:,:-24]
    poses_TKC = poses.reshape((len(poses),-1,3))
    smoothnet_poses8 = model(poses_TKC, type='rot')  
    poses_org[:,3:-24] = smoothnet_poses8.reshape((len(smoothnet_poses8),-1))[:,3:]
    poses_org[:,:3] = smooth_sequence_gaussian(poses_org[:,:3], sigma=1)
    poses_org[:,-4:] = smooth_sequence_gaussian(poses_org[:,-4:], sigma=1)
  
    restore_and_save_poses(poses_org, smplx_data_list,save_path)

def process_video_chunk(rank, video_paths, model):
    for video_path in tqdm(video_paths, desc=f'Processing video chunk {rank}'):
        if 'render' in str(video_path):
            continue
            
        basename = os.path.basename(video_path).split('.')[0]
        file_path = os.path.join(os.path.dirname(video_path), 'smplx', 'clip.pkl')
        save_path = os.path.join(os.path.dirname(video_path), 'smooth_smplx.pkl')

        if os.path.exists(save_path):
            print('save path exists')
            continue

        emoca_path = os.path.join(os.path.dirname(video_path), 'bbox.pkl')
        is_emoca_exists = True
        if not os.path.exists(emoca_path):
            print('bbox not exists')
            is_emoca_exists = False

        if is_emoca_exists:
            with open(emoca_path, 'rb') as file:
                sizes = pickle.load(file)
                centers = pickle.load(file)

            emoca_centers_sizes = []
            for i in range(len(centers)):
                centers[i][0][0] -=  sizes[i][0]/2
                centers[i][0][1] -=  sizes[i][0]/2
                centers[i] = np.append(centers[i][0],[ sizes[i][0],sizes[i][0]])
                emoca_centers_sizes.append(centers[i])

        print('file_path',file_path)
        if not os.path.exists(file_path):
            shutil.rmtree(os.path.dirname(video_path))
            print('rmdir',os.path.dirname(video_path))
            continue

        with open(file_path,'rb') as f:
            smplx_data = pickle.load(f)

        if is_emoca_exists:
            try:
                all_frames_data_one, drop_video = match_emoca_to_smplx(emoca_centers_sizes, smplx_data,is_emoca_exists = True)
            except:
                print('match_emoca_to_smplx error')
                shutil.rmtree(os.path.dirname(video_path))
                print('rmdir',os.path.dirname(video_path))
                continue
        else:
            all_frames_data_one = smplx_data

        if drop_video:
            continue

        smooth(all_frames_data_one, save_path, model)

if __name__ == '__main__':
    model = SmoothNetFilter(
            window_size=8,
            output_size=8,
            checkpoint='./smoothnet_windowsize8.pth.tar')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/amax/zyude/human_video_data_all/results_11_14')
    args = parser.parse_args()
    root_folder = args.root_path
    
    # Get all video paths
    video_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('clip.mp4'):
                video_paths.append(os.path.join(dirpath, filename))

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f'Found {num_gpus} GPUs')

    # Split videos among GPUs
    chunks = np.array_split(video_paths, num_gpus)
    
    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)
    processes = []

    # Start processes
    for rank in range(num_gpus):
        p = Process(target=process_video_chunk, 
                   args=(rank, chunks[rank], model))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()
