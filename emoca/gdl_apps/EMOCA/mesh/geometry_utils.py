import numpy as np
import torch
from torch_scatter import scatter
from easydict import EasyDict
from collections import defaultdict
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle
from tqdm import tqdm

def pose_vec_to_matrix(pose_vec):
    """
    Args:
        pose_vec [7]: (r; t; scale)
    """
    T = torch.eye(4).to(pose_vec)
    R = axis_angle_to_matrix(pose_vec[:3])
    T[:3, 3] = pose_vec[3:6]
    T[:3, :3] = R * pose_vec[-1]
    return T

def pose_matrix_to_vec(T):
    """
    Args:
        pose_vec [7]: (r; t; scale)
    """
    pose_vec = torch.zeros(7).to(T)
    pose_vec[3:6] = T[:3, 3]
    scale = ((T[:3, :3] @ T[:3, :3].T).trace() / 3).sqrt()
    R = T[:3, :3] / scale
    pose_vec[-1] = scale
    pose_vec[:3] = matrix_to_axis_angle(R)
    
    return pose_vec

def partial_load_obj(filename):
    vertices = []
    
    with open(filename, "r") as fin:
        for line in fin.readlines():
            tokens = line.strip().split(" ")
            if tokens[0] == "v":
                v = np.array([float(f) for f in tokens[1:]])
                vertices.append(v)
                if len(vertices) == 1000:
                    break

    return EasyDict(dict(pos=torch.from_numpy(np.array(vertices, dtype=np.float32))))

def estimate_normals_v2(v, triangles):
    face_nor = torch.cross(v[triangles[:, 0]] - v[triangles[:, 1]],
                           v[triangles[:, 2]] - v[triangles[:, 0]])

    vertex_nor = scatter(face_nor, triangles[:, 0], dim=0, dim_size=v.shape[0], reduce='sum')
    vertex_nor += scatter(face_nor, triangles[:, 1], dim=0, dim_size=v.shape[0], reduce='sum')
    vertex_nor += scatter(face_nor, triangles[:, 2], dim=0, dim_size=v.shape[0], reduce='sum')

    vertex_deg = scatter(torch.ones_like(face_nor), triangles[:, 0], dim=0, dim_size=v.shape[0], reduce='sum')
    vertex_deg += scatter(torch.ones_like(face_nor), triangles[:, 1], dim=0, dim_size=v.shape[0], reduce='sum')
    vertex_deg += scatter(torch.ones_like(face_nor), triangles[:, 2], dim=0, dim_size=v.shape[0], reduce='sum')

    vertex_nor = vertex_nor / vertex_deg
    vertex_nor = vertex_nor / vertex_nor.norm(p=2, dim=-1, keepdim=True)
    
    return -vertex_nor

def estimate_normals(v, nor):
    """
    Args:
        v [N, 3] point locations
        nor [N, 3] unsmooth normal vectors, but could be used for orientation
    Returns:
        
    """
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v).float().cuda()
    if isinstance(nor, np.ndarray):
        nor = torch.from_numpy(nor).float().cuda()

    from pytorch3d.ops import knn_points
    ret = knn_points(v[None, ...], v[None, ...], K=20, return_nn=True)
    
    #import polyscope as ps; ps.init()
    #ps_c = ps.register_point_cloud('point cloud', v.detach().cpu(), radius=1e-3)
    #ps_c.add_vector_quantity('nor', nor.detach().cpu(), radius=1e-3)
    #ps.show()
    
    #import ipdb; ipdb.set_trace()

    nor = torch.nn.Parameter(nor, requires_grad=True)
    idx = ret.idx[0]
    optimizer = torch.optim.AdamW([nor], lr=1e-3)

    count_down = 5
    last_loss = 1e10
    for itr in range(2500):
        optimizer.zero_grad()
        loss = (nor[:, None, :] - nor[idx]).square().mean()
        loss.backward()
        optimizer.step()
        nor.data = nor.data / nor.data.norm(p=2, dim=-1, keepdim=True)
        if loss.item() > last_loss - 1e4:
            count_down -= 1
        else:
            count_down = 5
        last_loss = loss.item()
        print(f'itr={itr}, loss={loss.item()}')

    #ps_c.add_vector_quantity('nor', nor.detach().cpu(), radius=1e-3)
    #ps.show()
    return nor.detach().cpu().numpy()

    ## [N, K, 3]
    #diff = ret.knn[0] - v[:, None, :]
    #
    ## [N, 3, K] @ [N, K, 3]
    #cov = diff.transpose(1, 2) @ diff
    #
    ## Pytorch Version is not working
    #eigvals, eigvecs = np.linalg.eigh(cov.detach().cpu().numpy())

    ## [N, 3]
    #normals = torch.from_numpy(eigvecs[..., 0]).to(v)

    #signs = (normals * nor).sum(-1).sign()

    #normals = normals * signs[..., None]
    #ps_c.add_vector_quantity('nor', nor.detach().cpu(), radius=1e-3)
    #ps.show()


def extract_faces_and_edges(mesh):
    edges = []
    faces = []
    for f in mesh.faces:
        new_f = [mesh.vertex_pos_indices[fi].item() for fi in f]
        for i in range(len(new_f)):
            if new_f[i-1] < new_f[i]:
                edges.append([new_f[i-1], new_f[i]])
        faces.append(new_f)
    edges = torch.from_numpy(np.array(edges, dtype=np.int32).reshape(-1, 2))

    return faces, edges


def transform(mesh, T):
    new_mesh = EasyDict(mesh)
    new_mesh.pos = (mesh.pos.double() @ T[:3, :3].T + T[:3, 3]).float()
    new_mesh.nor = (mesh.nor.double() @ T[:3, :3].T).float()

    return new_mesh


def barycentric_to_pos(v, f, b):
    """Translate bary-centric coordinates into point positions on tri mesh.
    Args:
        v [N, 3]
        f [M, 3]
        b [M, 3]
    Returns:
        pos [M, 3]
    """
    # [M, 3, 3]
    face_vertices = v[f]
    pos = (face_vertices * b[..., None]).sum(1)
    return pos


def load_and_convert_to_tri_mesh(filename, convert=True):
    """Given an obj file, read a triangular mesh with the following properties:
    - pos (positions) [attribute]
    - uv (texture coordinates) [attribute]
    - nor (normals) [attribute]
    - faces lists of vertex (not attribute) indices
    - vertex_pos_indices (each vertex associate w. one pos attribute)
    - vertex_uv_indices (each vertex associate w. one uv attribute)
    - vertex_nor_indices (each vertex associate w. one nor attribute)
    This function handles
    - duplicated vertices, one vertex can have multiple uv coordinates
    - inconsistent number of uv coordinates and vertices/normals
    - keep original pos/uv/nor attributes for subdivision purposes

    Args:
        dict.pos
        dict.uv
        dict.nor
        dict.faces
        dict.vertex_pos_indices
        dict.vertex_uv_indices
        dict.vertex_nor_indices
    """

    vertices = []
    normals = []
    texture_coords = []
    triplet_map = dict()
    
    faces = []
    triplets = []
    with open(filename, "r") as fin:
        for line in fin.readlines():
            tokens = line.strip().split(" ")
            if tokens[0] == "v":
                v = np.array([float(f) for f in tokens[1:4]])
                vertices.append(v)
            if tokens[0] == "vt":
                tex = np.array([float(f) for f in tokens[1:]])
                texture_coords.append(tex)
            if tokens[0] == "vn":
                nor = np.array([float(f) for f in tokens[1:]])
                normals.append(nor)
            if tokens[0] == "f":
                f = []
                for i, token in enumerate(tokens[1:]):
                    triplet = tuple([int(j)-1 for j in token.split('/')])
                    
                    if triplet not in triplet_map:
                        num_unique_triplets = len(triplet_map)
                        triplet_map[triplet] = num_unique_triplets
                        triplets.append(np.array(triplet, dtype=np.int32))

                    unique_triplet_id = triplet_map[triplet]

                    f.append(unique_triplet_id)
                if len(tokens[1:]) == 4 and convert:
                    faces.append([f[0], f[1], f[2]])
                    faces.append([f[2], f[3], f[0]])
                else:
                    faces.append(f)

        triplets = np.array(triplets, dtype=np.int32)
        
        if convert:
            faces = np.array(faces, dtype=np.int32)
        vertices = np.array(vertices)
        texture_coords = np.array(texture_coords)
        if texture_coords.shape[0] == 0:
            texture_coords = texture_coords.reshape(-1, 2)
        if triplets.shape[-1] > 2:
            normals = np.array(normals)
        if triplets.shape[-1] == 2:
            triplets = np.concatenate([triplets, triplets[:, :1]], axis=-1)
        
    if len(normals) == 0 and convert:
        v = np.array(vertices)
        pos_indices = triplets[:, 0]
        num_pos = pos_indices.max() + 1
        triangles = np.array(faces)
        face_nor = np.cross(v[pos_indices[triangles[:, 0]]] - v[pos_indices[triangles[:, 1]]],
                            v[pos_indices[triangles[:, 2]]] - v[pos_indices[triangles[:, 0]]])
        face_nor = torch.from_numpy(face_nor)
        triangles = torch.from_numpy(triangles).long()
        triangles = torch.from_numpy(triplets[:, 0]).long()[triangles]
    
        vertex_nor = scatter(face_nor, triangles[:, 0], dim=0, dim_size=v.shape[0], reduce='sum')
        vertex_nor += scatter(face_nor, triangles[:, 1], dim=0, dim_size=v.shape[0], reduce='sum')
        vertex_nor += scatter(face_nor, triangles[:, 2], dim=0, dim_size=v.shape[0], reduce='sum')
    
        vertex_deg = scatter(torch.ones_like(face_nor), triangles[:, 0], dim=0, dim_size=v.shape[0], reduce='sum')
        vertex_deg += scatter(torch.ones_like(face_nor), triangles[:, 1], dim=0, dim_size=v.shape[0], reduce='sum')
        vertex_deg += scatter(torch.ones_like(face_nor), triangles[:, 2], dim=0, dim_size=v.shape[0], reduce='sum')
    
        vertex_nor = vertex_nor / vertex_deg
        vertex_nor = vertex_nor / vertex_nor.norm(p=2, dim=-1, keepdim=True)
    
        normals = vertex_nor.numpy()
            
        normal_norms = np.linalg.norm(normals, ord=2, axis=-1)
        normals[np.isnan(normals[:, 0]), 1:] = 0.0
        normals[np.isnan(normals[:, 0]), 0] = 1.0
    
    return EasyDict(dict(
        pos = torch.tensor(np.array(vertices)),
        uv = torch.tensor(np.array(texture_coords))[:, :2],
        nor = torch.tensor(np.array(normals)),
        faces = faces,
        vertex_pos_indices = torch.tensor(triplets[:, 0]),
        vertex_uv_indices = torch.tensor(triplets[:, 1]),
        vertex_nor_indices = torch.tensor(triplets[:, 2]),
    ))

def subdivision(mesh, rigid=False):
    """
    Args:
        vertices [N, 3]
        faces: list of list of vertex ids

    Returns:
        new_vertices [N_2, 3]
        new_faces: list of list of vertex ids
    """

    # compute final topology
    new_mesh = EasyDict(dict())
    for key in ["pos", "uv", "nor"]:
        if not hasattr(mesh, key):
            continue
        new_attr, faces, new_vertex_attr_indices = catmull_clark(mesh[key], mesh.faces, mesh[f"vertex_{key}_indices"], key, rigid=rigid)
        new_mesh[f"vertex_{key}_indices"] = new_vertex_attr_indices
        if key == 'nor':
            new_mesh[key] = new_attr / new_attr.norm(p=2, dim=-1, keepdim=True)
        else:
            new_mesh[key] = new_attr
        if key == 'uv':
            new_mesh.faces = faces
    
    return new_mesh

def catmull_clark(attrs, vertex_faces, vertex_attr_indices, key, rigid=False):
    """
    Args:
        attrs [N, 3]: attributes such as (pos, uv, nor)
        faces: list of list of vertex ids
        vertex_attr_indices [V]: each vertex associate with one attribute, vertices may share attributes
    Returns:
        new_vertices [N2, 3]
        new_faces [F2, 4]
    """
    # graphs of vertex and attribute
    V, A = construct_graphs(attrs, vertex_faces, vertex_attr_indices, key, rigid=rigid)

    A.compute_face_points()

    #ps.register_point_cloud("face_points", A.face_points, radius=1e-3)

    A.compute_edge_points()
    
    #ps.register_point_cloud("edge_points", A.edge_points, radius=1e-3)

    A.compute_FR()
    
    #ps.register_point_cloud("F", A.F, radius=1e-3)
    #ps.register_point_cloud("R", A.R, radius=1e-3)

    A.update_attrs()
    
    #p = ps.register_point_cloud("updated", A.attrs, radius=1e-3)
    #p.add_scalar_quantity("boundary", A.boundary_mask)

    new_vertex_attr_indices = transfer_update(A, V, vertex_attr_indices)

    A_new_attrs, A_new_faces = A.construct_new_mesh()
    new_attrs, new_faces = V.construct_new_mesh()

    assert (new_attrs - A_new_attrs[new_vertex_attr_indices]).abs().max() < 1e-5
    assert (new_attrs - A_new_attrs[new_vertex_attr_indices]).abs().max() < 1e-5

    #for f, fa in zip(new_faces, A_new_faces):
    #    assert len(f) == len(fa)

    #    for j in range(len(f)):
    #        assert new_vertex_attr_indices[f[j]] == fa[j]

    return A_new_attrs, new_faces, new_vertex_attr_indices

def construct_graphs(attrs, v_faces, vertex_attr_indices, key, rigid=False):
    num_vertices = vertex_attr_indices.shape[0]
    V = Graph(attrs[vertex_attr_indices], num_vertices, v_faces, vertex_attr_indices, key, rigid=rigid)

    a_faces = []
    for f in v_faces:
        a_faces.append([vertex_attr_indices[vi].item() for vi in f])

    num_attrs = vertex_attr_indices.max().item() + 1
    A = Graph(attrs, num_attrs, a_faces, torch.arange(attrs.shape[0]).to(vertex_attr_indices), key, rigid=rigid)

    return V, A

class Graph:
    def __init__(self, attrs, num_nodes, faces, attr_uids, attr_name, rigid=False):
        self.attr_name = attr_name
        self.attrs = attrs
        self.num_nodes = num_nodes
        self.faces = faces
        self.attr_uids = attr_uids
        self.rigid = rigid

        self.deg_e = torch.zeros(self.num_nodes, dtype=torch.int32)
        self.deg_f = torch.zeros(self.num_nodes, dtype=torch.int32)

        num_edges = 0
        self.edge_faces, edges = [], []
        self.edge_index = dict()

        edge_adj_face_count = []

        for i, f in enumerate(faces):
            edge_face = []
            for j in range(len(f)):
                n0, n1 = f[j], f[(j+1)%len(f)]
                key = (n0, n1)

                self.deg_f[n0] += 1

                if key in self.edge_index:
                    e_index = self.edge_index[key]
                    edge_adj_face_count[e_index] += 1
                else:
                    e_index = num_edges
                    self.edge_index[(n0, n1)] = e_index
                    self.edge_index[(n1, n0)] = e_index
                    self.deg_e[n0] += 1
                    self.deg_e[n1] += 1
                    edge_adj_face_count.append(1)
                    num_edges += 1
                    edges.append([n0, n1])

                edge_face.append(e_index)
            self.edge_faces.append(edge_face)
        
        self.edges = edges
        self.num_edges = num_edges
        self.e0, self.e1 = torch.tensor(self.edges).reshape(-1, 2).T

        self.boundary_mask = self.deg_e != self.deg_f
        edge_adj_face_count = torch.tensor(edge_adj_face_count).to(attr_uids)
        self.boundary_edge_mask = edge_adj_face_count < 2
        self.corner_mask = (self.deg_e <= 2) | self.rigid

    def compute_face_points(self):
        face_points = []
        num_attrs = self.attrs.shape[0]
        for i, f in enumerate(self.faces):
            face_point = self.attrs[f].mean(0)
            face_points.append(face_point)
        self.face_points = torch.stack(face_points, dim=0).reshape(-1, self.attrs.shape[-1])

    def compute_edge_points(self):
        edge_points_adj = defaultdict(list)
        for i, f in enumerate(self.faces):
            for j in range(len(f)):
                v0, v1 = f[j], f[(j+1)%len(f)]

                key = (min(v0, v1), max(v0, v1))

                edge_points_adj[key].append(self.face_points[i])

        self.edge_points = torch.zeros(self.num_edges, self.attrs.shape[-1]).to(self.face_points)

        for key, val in edge_points_adj.items():
            v0, v1 = key

            mid_edge = (self.attrs[v0] + self.attrs[v1]) / 2
            if (len(val) == 2) and (not self.rigid):
                edge_point = torch.stack(val, dim=0).mean(0) * 0.5 + mid_edge * 0.5
            else:
                edge_point = mid_edge

            e_index = self.edge_index[key]
            self.edge_points[e_index] = edge_point
    
    def compute_FR(self):
        # for point P, take the average (F) of all n (recently created) face points for faces touching P
        # take the average (R) of all n edge midpoints for original edges touching P
        self.F = self.attrs.new_zeros((self.num_nodes, self.attrs.shape[-1]))
        self.R = self.attrs.new_zeros((self.num_nodes, self.attrs.shape[-1]))

        for i, f in enumerate(self.faces):
            for j in range(len(f)):
                n0, n1 = f[j], f[(j+1)%len(f)]

                self.F[n0] += self.face_points[i]

                mid_edge = (self.attrs[n0] + self.attrs[n1])/2
                self.R[n0] += mid_edge
                self.R[n1] += mid_edge
        
        self.F = self.F / self.deg_f[:, None]
        self.R = self.R / 2 / self.deg_e[:, None]

    def update_attrs(self):
        e0_b = self.e0[self.boundary_edge_mask]
        e1_b = self.e1[self.boundary_edge_mask]

        new_attrs = (self.F + 2 * self.R + (self.deg_e - 3)[:, None] * self.attrs) / self.deg_e[:, None]

        new_attrs[self.boundary_mask] = self.attrs[self.boundary_mask] * 0.75 \
                                        + 0.125 * scatter(self.attrs[e0_b], e1_b, dim=0, dim_size=self.num_nodes, reduce="sum")[self.boundary_mask] \
                                        + 0.125 * scatter(self.attrs[e1_b], e0_b, dim=0, dim_size=self.num_nodes, reduce="sum")[self.boundary_mask]

        new_attrs[self.corner_mask] = self.attrs[self.corner_mask]

        self.attrs = new_attrs

    def construct_new_mesh(self):
        new_attrs = torch.cat([self.attrs, self.edge_points, self.face_points])
        new_faces = []
        
        face_offset = 0
        for i, f in enumerate(self.faces):
            for j in range(len(f)):
                v0, v1, v2 = f[j], f[(j+1)%len(f)], f[(j+2)%len(f)]

                new_faces.append([i+self.num_nodes+self.num_edges,
                                  self.edge_index[(v0, v1)]+self.num_nodes,
                                  v1,
                                  self.edge_index[(v1, v2)]+self.num_nodes]
                                )
        
        return new_attrs, new_faces

def transfer_update(A, V, vertex_attr_indices):
    V.attrs = A.attrs[vertex_attr_indices].clone()
    V.face_points = A.face_points.clone()
    V.edge_points = A.edge_points.new_zeros(V.num_edges, V.attrs.shape[-1])

    edge_attr_indices = vertex_attr_indices.new_zeros(V.num_edges)
    for v0, v1 in V.edges:
        a0 = vertex_attr_indices[v0].item()
        a1 = vertex_attr_indices[v1].item()
        edge_index_v = V.edge_index[(v0, v1)]
        edge_index_a = A.edge_index[(a0, a1)]
        V.edge_points[edge_index_v] = A.edge_points[edge_index_a]
        edge_attr_indices[edge_index_v] = edge_index_a + A.num_nodes

    face_attr_indices = torch.arange(len(V.faces)) + A.num_edges + A.num_nodes

    new_vertex_attr_indices = torch.cat([vertex_attr_indices, edge_attr_indices, face_attr_indices], dim=0)

    return new_vertex_attr_indices

def load_obj(filename, convert_face_to_triangle=False, points_only=False):
    """Given an obj file, read a triangular mesh with the following properties:
    - pos (positions) [attribute]
    - uv (texture coordinates) [attribute]
    - nor (normals) [attribute]
    - faces lists of vertex (not attribute) indices
    - vertex_pos_indices (each vertex associate w. one pos attribute)
    - vertex_uv_indices (each vertex associate w. one uv attribute)
    - vertex_nor_indices (each vertex associate w. one nor attribute)
    This function handles
    - duplicated vertices, one vertex can have multiple uv coordinates
    - inconsistent number of uv coordinates and vertices/normals
    - keep original pos/uv/nor attributes for subdivision purposes

    Args:
        dict.pos
        dict.uv
        dict.nor
        dict.faces
        dict.vertex_pos_indices
        dict.vertex_uv_indices
        dict.vertex_nor_indices
    """

    vertices = []
    normals = []
    texture_coords = []
    triplet_map = dict()
    
    faces = []
    triplets = []
    with open(filename, "r") as fin:
        for line in fin.readlines():
            tokens = line.strip().split(" ")
            if tokens[0] == "v":
                v = np.array([float(f) for f in tokens[1:4]])
                vertices.append(v)
            if tokens[0] == "vt":
                tex = np.array([float(f) for f in tokens[1:]])
                texture_coords.append(tex)
            if tokens[0] == "vn":
                nor = np.array([float(f) for f in tokens[1:]])
                normals.append(nor)
            if tokens[0] == "f":
                f = []
                for i, token in enumerate(tokens[1:]):
                    triplet = tuple([int(j)-1 for j in token.split('/')])
                    
                    if triplet not in triplet_map:
                        num_unique_triplets = len(triplet_map)
                        triplet_map[triplet] = num_unique_triplets
                        triplets.append(np.array(triplet, dtype=np.int32))

                    unique_triplet_id = triplet_map[triplet]

                    f.append(unique_triplet_id)
                if convert_face_to_triangle and len(tokens[1:]) == 4:
                    faces.append([f[0], f[1], f[2]])
                    faces.append([f[2], f[3], f[0]])
                else:
                    faces.append(f)

        triplets = np.array(triplets, dtype=np.int32)
        
        if convert_face_to_triangle:
            faces = np.array(faces, dtype=np.int32)
        vertices = np.array(vertices)#[triplets[:, 0]]
        texture_coords = np.array(texture_coords) #[triplets[:, 1]]
        if texture_coords.shape[0] == 0:
            texture_coords = texture_coords.reshape(-1, 2)
        if triplets.shape[-1] > 2:
            normals = np.array(normals)#[triplets[:, 2]]
        if triplets.shape[-1] == 2:
            triplets = np.concatenate([triplets, triplets[:, :1]], axis=-1)
        
    if points_only:
        return EasyDict(dict(
            pos = torch.tensor(np.array(vertices)),
            faces = faces,
            vertex_pos_indices = torch.tensor(triplets[:, 0]),
        ))
    else:
        if len(normals) == 0:
            v = np.array(vertices)
            pos_indices = triplets[:, 0]
            num_pos = pos_indices.max() + 1
            try:
                triangles = np.array(faces)
                face_nor = np.cross(v[pos_indices[triangles[:, 0]]] - v[pos_indices[triangles[:, 1]]],
                                    v[pos_indices[triangles[:, 2]]] - v[pos_indices[triangles[:, 0]]])
                face_nor = torch.from_numpy(face_nor)
                triangles = torch.from_numpy(triangles).long()
                triangles = torch.from_numpy(triplets[:, 0]).long()[triangles]

                vertex_nor = scatter(face_nor, triangles[:, 0], dim=0, dim_size=v.shape[0], reduce='sum')
                vertex_nor += scatter(face_nor, triangles[:, 1], dim=0, dim_size=v.shape[0], reduce='sum')
                vertex_nor += scatter(face_nor, triangles[:, 2], dim=0, dim_size=v.shape[0], reduce='sum')

                vertex_deg = scatter(torch.ones_like(face_nor), triangles[:, 0], dim=0, dim_size=v.shape[0], reduce='sum')
                vertex_deg += scatter(torch.ones_like(face_nor), triangles[:, 1], dim=0, dim_size=v.shape[0], reduce='sum')
                vertex_deg += scatter(torch.ones_like(face_nor), triangles[:, 2], dim=0, dim_size=v.shape[0], reduce='sum')

                vertex_nor = vertex_nor / vertex_deg
                vertex_nor = vertex_nor / vertex_nor.norm(p=2, dim=-1, keepdim=True)

                normals = vertex_nor.numpy()
            except Exception as e:
                print(f"normal estimation v0 failed: {e}")
                print("re-trying estimating normals")
                normal_adj = [[] for i in range(num_pos)]
                for f in faces:
                    l = len(f)
                    for i in range(l):
                        f0 = pos_indices[f[i]]
                        f1 = pos_indices[f[(i+1)%l]]
                        f2 = pos_indices[f[(i+2)%l]]
                        face_nor = np.cross(v[f0] - v[f1], v[f2] - v[f0])
                        normal_adj[f0].append(face_nor)
                        normal_adj[f1].append(face_nor)
                        normal_adj[f2].append(face_nor)

                normals = []
                for i in range(num_pos):
                    ni = np.array(normal_adj[i]).mean(0)
                    normals.append(ni / np.linalg.norm(ni, ord=2))
                normals = np.array(normals)
                
            normal_norms = np.linalg.norm(normals, ord=2, axis=-1)
            normals[np.isnan(normals[:, 0]), 1:] = 0.0
            normals[np.isnan(normals[:, 0]), 0] = 1.0

        return EasyDict(dict(
            pos = torch.tensor(np.array(vertices)),
            uv = torch.tensor(np.array(texture_coords))[:, :2],
            nor = torch.tensor(np.array(normals)),
            faces = faces,
            vertex_pos_indices = torch.tensor(triplets[:, 0]),
            vertex_uv_indices = torch.tensor(triplets[:, 1]),
            vertex_nor_indices = torch.tensor(triplets[:, 2]),
        ))

def convert_face_to_triangle(faces):
    new_faces = []
    for f in faces:
        l = len(f)
        for i in range(l-2):
            new_f = [f[0], f[(i+1)%l], f[(i+2)%l]]
            new_faces.append(new_f)
    faces = torch.from_numpy(np.array(new_faces, dtype=np.int32))
    return faces

def save_obj(filename, mesh):
    """Given an obj file, write a triangular mesh with the following properties:
    - pos (positions) [attribute]
    - uv (texture coordinates) [attribute]
    - nor (normals) [attribute]
    - faces lists of vertex (not attribute) indices
    - vertex_pos_indices (each vertex associate w. one pos attribute)
    - vertex_uv_indices (each vertex associate w. one uv attribute)
    - vertex_nor_indices (each vertex associate w. one nor attribute)
    This function handles
    - duplicated vertices, one vertex can have multiple uv coordinates
    - inconsistent number of uv coordinates and vertices/normals
    - keep original pos/uv/nor attributes for subdivision purposes

    Args:
        dict.pos
        dict.uv
        dict.nor
        dict.faces
        dict.vertex_pos_indices
        dict.vertex_uv_indices
        dict.vertex_nor_indices
    """

    with open(filename, "w") as fout:
        # write pos
        for i in tqdm(range(mesh.pos.shape[0])):
            p = mesh.pos[i]
            fout.write(f"v {p[0].item()} {p[1].item()} {p[2].item()}\n")

        # write uv
        if hasattr(mesh, 'uv'):
            for i in tqdm(range(mesh.uv.shape[0])):
                uv = mesh.uv[i]
                fout.write(f"vt {uv[0].item()} {uv[1].item()}\n")

        # write nor
        if hasattr(mesh, 'nor'):
            for i in tqdm(range(mesh.nor.shape[0])):
                p = mesh.nor[i]
                fout.write(f"vn {p[0].item()} {p[1].item()} {p[2].item()}\n")

        if isinstance(mesh.faces, np.ndarray):
            f0, f1, f2 = mesh.faces.T
            vpi0 = mesh.vertex_pos_indices[f0] + 1
            vpi1 = mesh.vertex_pos_indices[f1] + 1
            vpi2 = mesh.vertex_pos_indices[f2] + 1
            vpi = np.stack([vpi0, vpi1, vpi2], axis=-1)
            
            if hasattr(mesh, 'vertex_uv_indices'):
                vui0 = mesh.vertex_uv_indices[f0] + 1
                vui1 = mesh.vertex_uv_indices[f1] + 1
                vui2 = mesh.vertex_uv_indices[f2] + 1
                vui = np.stack([vui0, vui1, vui2], axis=-1)
            
            if hasattr(mesh, 'vertex_nor_indices'):
                vni0 = mesh.vertex_nor_indices[f0] + 1
                vni1 = mesh.vertex_nor_indices[f1] + 1
                vni2 = mesh.vertex_nor_indices[f2] + 1
                vni = np.stack([vni0, vni1, vni2], axis=-1)

            for i in tqdm(range(mesh.faces.shape[0])):
                fout.write('f')
                for j in range(3):
                    fstr = f' {vpi[i, j]}'
                    if hasattr(mesh, 'vertex_uv_indices'):
                        fstr += f'/{vui[i, j]}'
                    else:
                        fstr += "/"
                    if hasattr(mesh, 'vertex_nor_indices'):
                        fstr += f'/{vni[i, j]}'
                    fout.write(fstr)
                fout.write("\n")
        else:        
            for f in mesh.faces:
                fout.write("f")
                for fj in f:
                    fstr = f' {mesh.vertex_pos_indices[fj]+1}'
                    if hasattr(mesh, 'vertex_uv_indices'):
                        fstr += f'/{mesh.vertex_uv_indices[fj]+1}'
                    else:
                        fstr += "/"
                    if hasattr(mesh, 'vertex_nor_indices'):
                        fstr += f'/{mesh.vertex_nor_indices[fj]+1}'
                    fout.write(fstr)
                    #fout.write(f" {mesh.vertex_pos_indices[fj]+1}/{mesh.vertex_uv_indices[fj]+1}/{mesh.vertex_nor_indices[fj]+1}")
                fout.write("\n")
