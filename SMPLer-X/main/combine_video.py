import cv2

import os
root_folder = '/amax/zyude/results_10_6/'
for dirpath, _, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.endswith('.mp4'):
            
        
            video_path = os.path.join(dirpath, filename)
            print(video_path)
            if 'render' in str(video_path):
                continue
            
            render_dir = os.path.join(os.path.dirname(video_path), 'render')
            smooth_render_dir = os.path.join(os.path.dirname(video_path), 'smooth_render')
            render_dir_combine = os.path.join(os.path.dirname(video_path), 'combine_render')

            
            
            os.makedirs(render_dir_combine, exist_ok=True)
    
            
            basename = os.path.basename(video_path).split('.')[0]
            out_mp4 = os.path.join(render_dir_combine, basename + '.mp4')
        
            cap1_mp4 = os.path.join(smooth_render_dir, basename + '.mp4')
            cap2_mp4 = os.path.join(render_dir, basename + '.mp4')

            if not os.path.exists(cap1_mp4):
                break
            if not os.path.exists(cap2_mp4):
                break


            
            cap1 = cv2.VideoCapture(cap1_mp4)
            cap2 = cv2.VideoCapture(cap2_mp4)

            
            width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap1.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

        
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_mp4, fourcc, fps, (width * 2, height))

            
            while True:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()

                if not ret1 or not ret2:
                    break

            
                combined_frame = cv2.hconcat([frame1, frame2])

                
                out.write(combined_frame)

        
            cap1.release()
            cap2.release()
            out.release()

            print("save at",out_mp4)
            ffmpeg_command = f"ffmpeg -i {out_mp4}  -filter_complex hstack {out_mp4}"

       
            os.system(ffmpeg_command)
