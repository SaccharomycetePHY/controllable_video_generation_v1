import glob
import json
import os
meta_dir = '/amax/zyude/controllable_video_generation/SMPLer-X/clips/results/clip_5-82_3/meta'

# 使用00001_0.json作为覆盖文件
cover_json_file = os.path.join(meta_dir, '00001_0.json')

# 读取覆盖文件
with open(cover_json_file, 'r') as f:
    cover_json_data = json.load(f)

for idx in range(80):
    json_files = glob.glob(os.path.join(meta_dir, f'{idx+1:05d}_*.json'))
    if json_files:
        # 覆盖所有json文件
        for json_file in json_files:
            with open(json_file, 'w') as f:
                json.dump(cover_json_data, f)
                print(f'已用 {cover_json_file} 覆盖 {json_file}')
