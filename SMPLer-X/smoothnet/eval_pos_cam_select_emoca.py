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
# from find_emoca_ids import is_bbox_contained,find_best_match_in_frame,match_emoca_to_smplx

# from vis_points import optimize_points
import shutil


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
        
        # 获取当前帧的人脸bbox
        if frame_index < len(emoca_centers_sizes):
            emoca_bbox = emoca_centers_sizes[frame_index]
            draw_bbox(frame, emoca_bbox, (255, 0, 0), "Face")

        # 匹配到的 person_data
        if frame_index < len(all_frames_data_one) and all_frames_data_one[frame_index] is not None:
            matched_person_data = all_frames_data_one[frame_index]
            body_bbox = matched_person_data['bbox_mmdet']
            draw_bbox(frame, body_bbox, (0, 255, 0), "Matched Body")

        # 未匹配的 person_data
        if frame_index < len(all_frames_data):
            all_persons = all_frames_data[frame_index]
            # print('frame_index',frame_index)
            # print(len(all_frames_data[frame_index]))
     
            for person_data in all_persons:
                # print(all_persons,person_data)
                body_bbox = person_data['bbox_mmdet']
                # 如果当前 person_data 不是匹配到的
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


# def is_bbox_contained(face_bbox, body_bbox):
  
#     face_x, face_y, face_w, face_h = face_bbox
#     body_x, body_y, body_w, body_h = body_bbox
    
#     return (face_x >= body_x and 
#             face_y >= body_y and 
#             face_x + face_w <= body_x + body_w and 
#             face_y + face_h <= body_y + body_h)

def find_best_match_in_frame(emoca_bbox, frame_data):
   
    best_id = 0
    idx = 0 
    is_find = False
    for person_data in frame_data:
        smplx_bbox = person_data['bbox_mmdet']
        if is_bbox_contained(emoca_bbox, smplx_bbox):
            best_id = idx
            is_find = True
            break  # 找到第一个包含人脸bbox的人体bbox后立即返回
        idx += 1
    if not is_find:
        print('is find',is_find)


    return best_id

def match_emoca_to_smplx(emoca_centers_sizes, all_frames_data,is_emoca_exists = True):
    import copy
   
    all_frames_data_one = copy.deepcopy(all_frames_data)

    prev_best_person_id = None  

    for frame_index, frame_data in enumerate(all_frames_data):
        emoca_bbox = emoca_centers_sizes[frame_index]  # 当前帧的人脸bbox
        
       
        best_id = find_best_match_in_frame(emoca_bbox, frame_data)


        all_frames_data_one[frame_index][0] = all_frames_data_one[frame_index][best_id]

    return all_frames_data_one





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
    poses = poses[:,:-24]
    poses_TKC = poses.reshape((len(poses),-1,3))
    smoothnet_poses8 = model(poses_TKC, type='rot')  
    poses_org[:,3:-24] = smoothnet_poses8.reshape((len(smoothnet_poses8),-1))[:,3:]
    poses_org[:,:3] = smooth_sequence_gaussian(poses_org[:,:3], sigma=1)
    poses_org[:,-4:] = smooth_sequence_gaussian(poses_org[:,-4:], sigma=1)
    # print(poses_org.shape[:])#(80, 192)
  
    restore_and_save_poses(poses_org, smplx_data_list,save_path)

if __name__ == '__main__':
    model = SmoothNetFilter(
            window_size=8,
            output_size=8,
            checkpoint='./smoothnet_windowsize8.pth.tar')

    root_folder = '/amax/zyude/human_video_data_all/data_one_person'
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('clip.mp4'):
                
            
                video_path = os.path.join(dirpath, filename)
                print(video_path)
                if 'render' in str(video_path):
                    continue
               
                # smplx_dir = os.path.join(os.path.dirname(video_path), 'smplx')

                
                # if not os.path.exists(smplx_dir):
                #     break

                # smooth_smplx_dir = os.path.join(os.path.dirname(video_path), 'smooth_smplx')
                # smooth_smplx_dir = os.path.join(os.path.dirname(video_path), 'clip.pkl')

                # os.makedirs(smooth_smplx_dir, exist_ok=True)


                basename = os.path.basename(video_path).split('.')[0]

                file_path = os.path.join(os.path.dirname(video_path), 'smplx', 'clip.pkl')

                save_path = os.path.join(os.path.dirname(video_path),  'smooth_smplx.pkl')
                if os.path.exists(save_path):
                    
                    print('save path exists')
                    continue
            
                # file_path = os.path.join(smplx_dir, basename + '.pkl')

                emoca_path = os.path.join(os.path.dirname(video_path), 'bbox.pkl')
                is_emoca_exists = True
                if not os.path.exists(emoca_path):
                    print('bbox not exists')
                    is_emoca_exists = False
                    # continue
                if is_emoca_exists:
                    with open(emoca_path, 'rb') as file:
                        # detection_fnames = pickle.load(file)
                        sizes = pickle.load(file)
                        centers = pickle.load(file)
                    # print(sizes,centers)
                    # exit()

                
                    emoca_centers_sizes = []
                    for i in range(len(centers)):
                        # print(centers[i])
                        # tmp = centers[i][0][0]
                        # centers[i][0][1] = centers[i][0][1]
                        # centers[i][0][1] = tmp
                        centers[i][0][0] -=  sizes[i][0]/2
                        centers[i][0][1] -=  sizes[i][0]/2
                        
                        centers[i] = np.append(centers[i][0],[ sizes[i][0],sizes[i][0]])
                    
                        # print('centers',centers[i])
                        emoca_centers_sizes.append(centers[i])

                print('file_path',file_path)
                if not os.path.exists(file_path):
                    shutil.rmtree(os.path.dirname(video_path))
                    print('rmdir',os.path.dirname(video_path))
                    continue
                with open(file_path,'rb') as f:

                    smplx_data = pickle.load(f)
                    # for i in range(len(smplx_data)):
                    #     print(len(smplx_data[i]))

                if is_emoca_exists:
                    try:
                        all_frames_data_one = match_emoca_to_smplx(emoca_centers_sizes, smplx_data,is_emoca_exists = True)
                    except:
                        print('match_emoca_to_smplx error')
                        shutil.rmtree(os.path.dirname(video_path))
                        print('rmdir',os.path.dirname(video_path))
                        continue
                else:
                    all_frames_data_one = smplx_data
                smooth(all_frames_data_one,save_path,model)
                # visualize_video_with_bbox(video_path, emoca_centers_sizes, smplx_data, all_frames_data_one, output_video_path='test.mp4')

                

              

  
    # with open('../SmoothNet-main/find_emoca_ids/clip_5_83_with_cam.pkl', 'rb') as f:
    #     input_data = pickle.load(f)
    # output_name = 'clip_5-82_smooth_cam_root_together_-1_1017'    
    # smooth(input_data,output_name,model)
