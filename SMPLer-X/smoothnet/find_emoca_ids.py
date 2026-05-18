import pickle
import numpy as np
import json
import os

import matplotlib.pyplot as plt
import cv2



def process_files(smplx_dir, meta_dir):
    all_frames = {}  # 字典，key 为 frame_id，value 是该帧所有人的数据列表

   
    for file in sorted(os.listdir(smplx_dir)):
        if file.endswith('.npz'):
            smplx_file_path = os.path.join(smplx_dir, file)
            
            # 解析文件名，获取帧号和人 ID
            frame_id, person_id = file.split('_')
            person_id = person_id.replace('.npz', '')

         
            meta_file_name = f"{frame_id}_{person_id}.json"
            meta_file_path = os.path.join(meta_dir, meta_file_name)
            
            if not os.path.exists(meta_file_path):
                print(f"Meta file not found: {meta_file_name}")
                continue

       
            smplx_data = np.load(smplx_file_path, allow_pickle=True)
            with open(meta_file_path, 'r') as f:
                meta_data = json.load(f)
            
            # 获取 'bbox_mmdet' 对应的值，并添加到 smplx 字典中
            if 'bbox_mmdet' in meta_data:
                smplx_data = dict(smplx_data)  # 转换为字典
                smplx_data['bbox_mmdet'] = meta_data['bbox_mmdet']  
            else:
                print(f"'bbox_mmdet' not found in: {meta_file_name}")
                continue

            # 创建当前帧的列表，如果之前不存在该帧则初始化
            if frame_id not in all_frames:
                all_frames[frame_id] = []

            # 添加该人的数据到当前帧的列表中
            person_data = {
                "person_id": int(person_id),
                "smplx_data": smplx_data
            }
            all_frames[frame_id].append(person_data)

    # 将字典转换为按帧顺序排列的列表
    sorted_all_frames = []
    for frame_id in sorted(all_frames.keys(), key=lambda x: int(x)):  # 按帧号排序
        sorted_all_frames.append(all_frames[frame_id])

    return sorted_all_frames

#匹配emoca

def is_bbox_contained(face_bbox, body_bbox):
  
    face_x, face_y, face_w, face_h = face_bbox
    body_x, body_y, body_w, body_h = body_bbox
    
    return (face_x >= body_x and 
            face_y >= body_y and 
            face_x + face_w <= body_x + body_w and 
            face_y + face_h <= body_y + body_h)

def find_best_match_in_frame(emoca_bbox, frame_data):
   
    best_person_data = None

    for person_data in frame_data:
        smplx_bbox = person_data['smplx_data']['bbox_mmdet']
        if is_bbox_contained(emoca_bbox, smplx_bbox):
            best_person_data = person_data
            break  # 找到第一个包含人脸bbox的人体bbox后立即返回

    return best_person_data

def match_emoca_to_smplx(emoca_centers_sizes, all_frames_data):
   
    all_frames_data_one = []

    prev_best_person_id = None  

    for frame_index, frame_data in enumerate(all_frames_data):
        emoca_bbox = emoca_centers_sizes[frame_index]  # 当前帧的人脸bbox
        
       
        best_person_data = find_best_match_in_frame(emoca_bbox, frame_data)

#      
#         if best_person_data is None and prev_best_person_id is not None:
#             for person_data in frame_data:
#                 if person_data['person_id'] == prev_best_person_id:
#                     best_person_data = person_data
#                     break


#         if best_person_data:
#             prev_best_person_id = best_person_data['person_id']
#         else:
#             prev_best_person_id = None

        all_frames_data_one.append(best_person_data)

    return all_frames_data_one

# all_frames_data_one = match_emoca_to_smplx(emoca_centers_sizes, all_frames_data)


# for frame_index, person_data in enumerate(all_frames_data_one):
#     if person_data:
#         print(f"Frame {frame_index + 1}: Matched person ID: {person_data['person_id']}",person_data['smplx_data']['bbox_mmdet'])
#     else:
#         print(f"Frame {frame_index + 1}: No match found")

        
#可视化


def draw_bbox(image, bbox, color, label):
 
    x, y, w, h = bbox
    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
    cv2.putText(image, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def visualize_video_with_bbox(video_path, emoca_centers_sizes, all_frames_data, all_frames_data_one, output_video_path):
 
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
            body_bbox = matched_person_data['smplx_data']['bbox_mmdet']
            draw_bbox(frame, body_bbox, (0, 255, 0), "Matched Body")

        # 未匹配的 person_data
        if frame_index < len(all_frames_data):
            all_persons = all_frames_data[frame_index]
            for person_data in all_persons:
                body_bbox = person_data['smplx_data']['bbox_mmdet']
                # 如果当前 person_data 不是匹配到的
                if matched_person_data is None or body_bbox != matched_person_data['smplx_data']['bbox_mmdet']:
                    draw_bbox(frame, body_bbox, (0, 0, 255), "Unmatched Body")

     
        out.write(frame)
        frame_index += 1

   
        if frame_index % 50 == 0:
            print(f"Processed {frame_index}/{frame_count} frames")
    

    cap.release()
    out.release()
    print(f"保存视频到: {output_video_path}")


if __name__ == '__main__':

        
    with open('clip_5-82_bbox.pkl', 'rb') as file:
        # detection_fnames = pickle.load(file)
        sizes = pickle.load(file)
        centers = pickle.load(file)


    # print(detection_fnames)




    emoca_centers_sizes = []
    for i in range(len(centers)):
        print(centers[i])
        # tmp = centers[i][0][0]
        # centers[i][0][1] = centers[i][0][1]
        # centers[i][0][1] = tmp
        centers[i][0][0] -=  sizes[i][0]/2
        centers[i][0][1] -=  sizes[i][0]/2
        
        centers[i] = np.append(centers[i][0],[ sizes[i][0],sizes[i][0] ])
    
        # print('centers',centers[i])
        emoca_centers_sizes.append(centers[i])

    
    print(emoca_centers_sizes)

    
    all_frames_data = process_files(smplx_dir='results/clip_5-82/smplx/', meta_dir='./results/clip_5-82/meta/')

    # print(len(smplerx_result))
    # print(len(emoca_centers_sizes))

    for frame_index, frame_data in enumerate(all_frames_data):
        print(emoca_centers_sizes[frame_index])
        for person_data in frame_data:
            print(f"Person ID: {person_data['person_id']}, SMPLX Data Keys: {person_data['smplx_data']['bbox_mmdet']}")
    all_frames_data_one = match_emoca_to_smplx(emoca_centers_sizes, all_frames_data)      
    video_path = "clip_5-82.mp4"  
    output_video_path = "output_video_with_bbox.mp4" 


    visualize_video_with_bbox(video_path, emoca_centers_sizes, all_frames_data, all_frames_data_one, output_video_path)

    with open('clip_5_83.pkl', 'wb') as f:
        pickle.dump(all_frames_data_one, f)

