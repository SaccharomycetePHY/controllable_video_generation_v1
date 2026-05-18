import smplx
import torch
import numpy as np
import trimesh

def load_smplx_parameters(npz_path):
  
    data = np.load(npz_path)

    transl = data['transl']
    global_orient = data['global_orient'] 
    body_pose = data['body_pose'] 
    left_hand_pose = data['left_hand_pose']  
    right_hand_pose = data['right_hand_pose'] 
    jaw_pose = data['jaw_pose'] 
    leye_pose = data['leye_pose'] 
    reye_pose = data['reye_pose'] 
    betas = data['betas']  
    expression = data['expression'] 

    return transl, global_orient, body_pose, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, betas, expression

def smplx_to_mesh(transl, global_orient, body_pose, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, betas, expression):
    # 加载 SMPL-X 模型
    model = smplx.create(model_path='/amax/zyude/controllable_video_generation/SMPLer-X/common/utils/human_model_files', model_type='smplx',
                         gender='neutral', use_pca=False)
    
#     model = smplx.build_layer(
#         '../../SMPLX_vis/human_model_files/', model_type='smplx',
#         gender='neutral', use_face_contour=False,
#         num_betas=10,
#         num_expression_coeffs=10,
#         ext='npz')
#     print(model.faces.shape[:])
   


    transl = torch.tensor(transl, dtype=torch.float32)
    global_orient = torch.tensor(global_orient, dtype=torch.float32)
    body_pose = torch.tensor(body_pose, dtype=torch.float32)
    left_hand_pose = torch.tensor(left_hand_pose, dtype=torch.float32)
    right_hand_pose = torch.tensor(right_hand_pose, dtype=torch.float32)
    jaw_pose = torch.tensor(jaw_pose, dtype=torch.float32)
    leye_pose = torch.tensor(leye_pose, dtype=torch.float32)
    reye_pose = torch.tensor(reye_pose, dtype=torch.float32)
    betas = torch.tensor(betas, dtype=torch.float32)
    expression = torch.tensor(expression, dtype=torch.float32)
    

    print(transl.shape[:])
    print(global_orient.shape[:])
    print(expression.shape[:])
    print(betas.shape[:])
    print(left_hand_pose.shape[:])
    # print(transl.shape[:])
    # print(transl.shape[:])


    # betas = torch.randn([1, model.num_betas], dtype=torch.float32).clamp(0, 0.1)
    # expression = torch.randn([1, model.num_expression_coeffs], dtype=torch.float32).clamp(0, 0.1)
    # body_pose = torch.randn([1, 21, 3], dtype=torch.float32).clamp(0, 0.4)
    # output = model(
    #     betas=betas, expression=expression, body_pose=body_pose, return_verts=True
    # )
    # transl = np.zeros((3,))  # 位移 (3,)
    # global_orient = np.zeros((3,))  # 全局旋转 (3,)
    # body_pose = np.zeros((21*3,))  # 身体姿态 (63,)
    # left_hand_pose = np.zeros((15*3,))  # 左手姿态 (45,)
    # right_hand_pose = np.zeros((15*3,))  # 右手姿态 (45,)
    # transl = torch.tensor(transl, dtype=torch.float32).unsqueeze(0)
    # global_orient = torch.tensor(global_orient, dtype=torch.float32).unsqueeze(0)
    # body_pose = torch.tensor(body_pose, dtype=torch.float32).unsqueeze(0)
    # left_hand_pose = torch.tensor(left_hand_pose, dtype=torch.float32).unsqueeze(0)
    # right_hand_pose = torch.tensor(right_hand_pose, dtype=torch.float32).unsqueeze(0)
    output = model(transl=transl, global_orient=global_orient, body_pose=body_pose,
                   left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
                  )


    vertices = output.vertices[0].detach().cpu().numpy().squeeze()
    faces = model.faces  

    return vertices, faces

def save_mesh(vertices, faces, output_path):

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(output_path)


npz_path = '/amax/zyude/controllable_video_generation/SMPLer-X/smoothnet/clip_203-311.npz'
# npz_path = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/danhui-240108120070/zyd-video_generation/DATA/smplx_output/clip_10040-10137/00001_0.npz' 
output_mesh_path = 'output_mesh4.obj' 


transl, global_orient, body_pose, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, betas, expression = load_smplx_parameters(npz_path)
print(len(transl))
import os

output_dir = '/amax/zyude/controllable_video_generation/SMPLer-X/demo/results/clip_203-311/mesh'
os.makedirs(output_dir, exist_ok=True)

for idx in range(len(transl)):
    vertices, faces = smplx_to_mesh(transl[idx], global_orient[idx], body_pose[idx].reshape((1,63)), 
                                    left_hand_pose[idx].reshape((1,45)), right_hand_pose[idx].reshape((1,45)), 
                                    jaw_pose[idx], leye_pose[idx], reye_pose[idx],
                                    betas[idx], expression[idx])
    
    output_mesh_path = os.path.join(output_dir, f'{idx+1:05d}_0.obj')
    save_mesh(vertices, faces, output_mesh_path)
    print(f'已保存mesh: {output_mesh_path}')
