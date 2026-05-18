import cv2
import torch
import numpy as np
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import time
from depth_anything_v2.dpt import DepthAnythingV2
import matplotlib
cmap = matplotlib.colormaps.get_cmap('Spectral_r')
print(torch.cuda.is_available())
print('end')

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# DEVICE = 'cpu'
import argparse


parser = argparse.ArgumentParser(description="add root_path.")
parser.add_argument('--root_path', type=str, required=True, help="Path to the root directory")
args = parser.parse_args()
root_path = args.root_path
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl'  # or 'vits', 'vitb', 'vitg'
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

# input_dir = '/amax/zyude/human_video_data/389988163'  
#input_dir = '/amax/zyude/human_video_data_all/data'
input_dir  =  root_path
# output_dir ='/amax/zyude/data/video_9_27_depth/' 

# input_dir = '/amax/zyude/data/video_10_6'  
# output_dir ='/amax/zyude/data/video_10_6_depth/' 

for root, _, files in os.walk(input_dir):
    for filename in files:
        if filename.startswith('clip') and filename.endswith('.mp4'):
            
            
            video_path = os.path.join(root, filename)
            print(video_path)
            cap = cv2.VideoCapture(video_path)

            save_path = os.path.join(os.path.dirname(video_path))

                
 #           os.makedirs(save_path, exist_ok=True)
            basename = os.path.basename(video_path).split('.')[0]
           
            out_path = os.path.join(save_path, 'depth.npy')
            if os.path.exists(out_path):
                print(f"File already exists at {out_path}")
                break 
            # out_filename = f'{os.path.splitext(filename)[0]}.npy'
            
   
            # relative_path = os.path.relpath(root, input_dir)
            # output_folder = os.path.join(output_dir, relative_path)
            # out_path = os.path.join(output_folder, out_filename)
            
        
            # os.makedirs(output_folder, exist_ok=True)
            # out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
            #                       (int(cap.get(3)), int(cap.get(4))))  # 宽度翻倍
            num = 0
            start = time.time()
            all_arry = []
            while cap.isOpened():
                num += 1
#                print(num)
                
                ret, frame = cap.read()
                if not ret:
                    break

                depth = model.infer_image(frame)  # HxW raw depth map in numpy
                # print((time.time()-start)/num)
            
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                # print(depth)
                # print(depth.shape[:])
                all_arry.append(depth)
                # exit()

                #depth_colored = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

                
                # out.write(depth)
#                 depth_norm = depth / depth.max()  # 归一化
#                 depth_colored = (depth_norm * 255).astype(np.uint8)
#                 depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_GRAY2BGR)

#               
#                 combined = np.hstack((frame, depth_colored))
#                 out.write(combined)
            all_arry = np.array(all_arry)
            np.save(out_path,all_arry)
            print(all_arry.shape[:])

            cap.release()
            # out.release()
            end = time.time()
            print((end-start)/num)
            # exit()

print("Processing complete.")
