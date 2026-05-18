import numpy as np
import sys
import glob
from tqdm import tqdm

import geometry_utils, sa_utils

import polyscope.imgui as psim

frame = 0
vertices_list = []

def bs_callback():
    global vertices_list, frame
    psim.PushItemWidth(150)

    psim.TextUnformatted("Animation of Meshes")
    changed, frame = psim.SliderFloat("animation", frame, v_min=0, v_max=800)
    frame = int(frame)
    
    if psim.Button("+"):
        frame += 1
        changed = True
    if psim.Button("-"):
        frame -= 1
        changed = True

    if changed:
        ps_blend = ps.get_surface_mesh('mesh')
        pos = vertices_list[frame]
        ps_blend.update_vertex_positions(pos)
        
        #marker_blend = neutral_marker + marker_bs * bs_w
        #ps_marker = ps.get_point_cloud('marker_blend')
        #ps_marker.update_point_positions(marker_blend)

if __name__ == '__main__':
    import polyscope as ps; ps.init(); ps.set_up_dir('y_up')

    mesh_files = sorted(glob.glob('*_000.obj'))

    vertex_list = []

    mesh0 = geometry_utils.load_and_convert_to_tri_mesh(mesh_files[0], convert=False)
    faces = mesh0.faces
    ps.register_surface_mesh('mesh', mesh0.pos[mesh0.vertex_pos_indices], faces)

    for i, mf in tqdm(enumerate(mesh_files[:800])):
        if sa_utils.exists(f'mesh_{i:03d}_000'):
            v = sa_utils.read_data(f'mesh_{i:03d}_000')
        else:
            mesh = geometry_utils.load_and_convert_to_tri_mesh(mf, convert=False)
            v = mesh.pos[mesh.vertex_pos_indices]
            sa_utils.write_data(v, f'mesh_{i:03d}_000')
        vertices_list.append(v)
    ps.set_user_callback(bs_callback)
    ps.show()

    #exp_id = input()
    #exp_id = int(exp_id)

    #subdiv = True
    #exp_name = 'marker_v0'
    #actor='wangsiran_test_0305'
    #standard = False

    #if standard:
    #    folder_name = 'standard_expressions'
    #else:
    #    folder_name = 'expressions'
    #
    #neutral_folder = f'Y:\xMovRDprojs\MultilinearModelFaceV5\{actor}\RefinedNeutral'
    #exp_folder = f'Y:\xMovRDprojs\MultilinearModelFaceV5\{actor}\{marker_v0}\phase_point_results\{folder_name}'
    #
    ## load meshes
    #if subdiv: 
    #    mesh1 = geometry_utils.load_and_convert_to_tri_mesh(f'{neutral_folder}\{actor}_scanmod_subdiv.obj', convert=False)
    #    mesh2 = geometry_utils.load_and_convert_to_tri_mesh(f'{exp_folder}\subdiv_exp_{exp_id:02d}.obj', convert=False)
    #else:
    #    mesh1 = geometry_utils.load_and_convert_to_tri_mesh(f'{neutral_folder}\{actor}_scanmod.obj', convert=False)
    #    mesh2 = geometry_utils.load_and_convert_to_tri_mesh(f'{exp_folder}\exp_{exp_id:02d}.obj', convert=False)

    #pos_faces, _ = geometry_utils.extract_faces_and_edges(mesh1)
    #
    #neutral = mesh1.pos
    #bs = mesh2.pos - mesh1.pos
    #
    ## load markers on mesh
    #neutral_marker = np.load(f'{exp_folder}/neutral_markers.npy')
    #exp_marker = np.load(f'{exp_folder}/exp_mesh_{exp_id:02d}_markers.npy')
    #marker_bs = exp_marker - neutral_marker
    #
    ## load scans
    #neutral_scan_v, neutral_scan_f = io_utils.load_scan_or_cache(f'{neutral_folder}/{actor}_tran_scan.obj', True, 'normal')
    #neutral_scan_v = neutral_scan_v[..., :3]
    #exp_scan.v, exp_scan.f = io_utils.load_scan_or_cache(f'{exp_folder}/scans/scan_{exp_id:02d}.obj', True, 'normal')
    #exp_scan.v = exp_scan.v[..., :3]

    ## load markers on scans
    #scan_marker = np.load(f'{exp_folder}/scan_markers/exp_scan_{exp_id:02d}_markers.npy')
    #
    ## visualization
    #ps_neutral = ps.register_surface_mesh(f'blend', neutral_mesh.pos, neutral_pos_faces, edge_width=1.0, edge_color=[0.3, 0.7, 0.3], color=[0.3, 0.3, 0.3]) #, color=[0.9, 0.55, 0.1])
    #ps_marker = ps.register_point_cloud(f'marker_blend', neutral_marker, radius=1e-3, color=[1, 1, 0]) #, color=[0.9, 0.55, 0.1])
    #ps_marker_target = ps.register_point_cloud(f'marker_target', scan_marker, radius=1e-3, color=[1, 0, 0]) #, color=[0.9, 0.55, 0.1])
    #
    #ps.set_user_callback(bs_callback)
    #ps.show()
