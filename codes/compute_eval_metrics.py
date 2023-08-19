import trimesh
from sklearn.neighbors import KDTree
import numpy as np
from multiprocessing import Pool
from glob import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', required=True, help='directory to the primitive meshes')
parser.add_argument('--pred_path', required=True, help='directory to the primitive meshes')
opt = parser.parse_args()


sample_num = 100000
f1_threshold = 0.003
ef1_radius = 0.004
ef1_dotproduct_threshold = 0.2
ef1_threshold = 0.005

def compute_angles(points, triangles):
    A = points[triangles[:, 0], :]  # triangle vertex A
    B = points[triangles[:, 1], :]  # triangle vertex B
    C = points[triangles[:, 2], :]  # triangle vertex C

    # calculate the angles
    l_AB = np.sqrt(np.sum((A - B) ** 2, axis=-1)) + 1e-10
    l_AC = np.sqrt(np.sum((A - C) ** 2, axis=-1)) + 1e-10
    l_BC = np.sqrt(np.sum((B - C) ** 2, axis=-1)) + 1e-10
    
    dot_A = np.sum((B - A) * (C - A), axis=-1) / (l_AB * l_AC)
    dot_B = np.sum((A - B) * (C - B), axis=-1) / (l_AB * l_BC)
    dot_C = np.sum((A - C) * (B - C), axis=-1) / (l_AC * l_BC)
    dot_A = np.clip(dot_A, -1., 1.)  # to avoid complex values
    dot_B = np.clip(dot_B, -1., 1.)
    dot_C = np.clip(dot_C, -1., 1.)
    angle_A = np.arccos(dot_A)  # [0, PI] in radians
    angle_B = np.arccos(dot_B)
    angle_C = np.arccos(dot_C)
    angles = np.stack([angle_A, angle_B, angle_C], axis=1)
    return (angles.reshape(-1))*180/np.pi


def normalize_diagonal(path):
    old_mesh = trimesh.load_mesh(path)
    old_mesh = old_mesh.as_open3d

    # old_mesh = o3d.io.read_triangle_mesh(path)
    vertices = np.asarray(old_mesh.vertices)
    triangles = np.asarray(old_mesh.triangles)

    #normalize diagonal=1
    x_max = np.max(vertices[:,0])
    y_max = np.max(vertices[:,1])
    z_max = np.max(vertices[:,2])
    x_min = np.min(vertices[:,0])
    y_min = np.min(vertices[:,1])
    z_min = np.min(vertices[:,2])
    x_mid = (x_max+x_min)/2
    y_mid = (y_max+y_min)/2
    z_mid = (z_max+z_min)/2
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    scale = np.sqrt(x_scale*x_scale + y_scale*y_scale + z_scale*z_scale)
    vertices[:,0] = (vertices[:,0]-x_mid)/scale
    vertices[:,1] = (vertices[:,1]-y_mid)/scale
    vertices[:,2] = (vertices[:,2]-z_mid)/scale
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    return new_mesh

def get_metrics(gt_path, pred_path):
    #load gt
    gt_mesh = normalize_diagonal(gt_path)
    gt_points, gt_indices = gt_mesh.sample(sample_num, return_index=True)
    gt_normals = gt_mesh.face_normals[gt_indices]
    #load pred
    pred_mesh = normalize_diagonal(pred_path)
    pred_points, pred_indices = pred_mesh.sample(sample_num, return_index=True)
    pred_normals = pred_mesh.face_normals[pred_indices]

    pred_vertices = pred_mesh.vertices
    pred_triangles = pred_mesh.faces
    pred_angles = compute_angles(pred_vertices, pred_triangles)

    small_angles = []
    for i in range(40):
        small_angles.append( np.mean( (pred_angles<i).astype(np.float32) ) )

    # cd and nc and f1
    # from gt to pred
    pred_tree = KDTree(pred_points)
    dist, inds = pred_tree.query(gt_points, k=1)
    recall = np.sum(dist < f1_threshold) / float(len(dist))
    gt2pred_mean_cd1 = np.mean(dist)
    dist = np.square(dist)
    gt2pred_mean_cd2 = np.mean(dist)
    neighbor_normals = pred_normals[np.squeeze(inds, axis=1)]
    dotproduct = np.abs(np.sum(gt_normals*neighbor_normals, axis=1))
    gt2pred_nc = np.mean(dotproduct)
    gt2pred_nr = np.mean(np.degrees(np.arccos(np.minimum(dotproduct,1.0))))

    gt2pred_na = []
    for i in range(90):
        gt2pred_na.append( np.mean( (dotproduct<np.cos(i/180.0*np.pi)).astype(np.float32) ) )

    # from pred to gt
    gt_tree = KDTree(gt_points)
    dist, inds = gt_tree.query(pred_points, k=1)
    precision = np.sum(dist<f1_threshold)/float(len(dist))
    pred2gt_mean_cd1 = np.mean(dist)
    dist = np.square(dist)
    pred2gt_mean_cd2 = np.mean(dist)
    neighbor_normals = gt_normals[np.squeeze(inds, axis=1)]
    dotproduct = np.abs(np.sum(pred_normals*neighbor_normals, axis=1))
    pred2gt_nc = np.mean(dotproduct)
    pred2gt_nr = np.mean(np.degrees(np.arccos(np.minimum(dotproduct,1.0))))

    pred2gt_na = []
    for i in range(90):
        pred2gt_na.append( np.mean( (dotproduct<np.cos(i/180.0*np.pi)).astype(np.float32) ) )

    cd1 = gt2pred_mean_cd1+pred2gt_mean_cd1
    cd2 = gt2pred_mean_cd2+pred2gt_mean_cd2
    nc = (gt2pred_nc+pred2gt_nc)/2
    nr = (gt2pred_nr+pred2gt_nr)/2
    if recall+precision > 0: f1 = 2 * recall * precision / (recall + precision)
    else: f1 = 0

    #sample gt edge points
    indslist = gt_tree.query_radius(gt_points, ef1_radius)
    flags = np.zeros([len(gt_points)],bool)
    for p in range(len(gt_points)):
        inds = indslist[p]
        if len(inds)>0:
            this_normals = gt_normals[p:p+1]
            neighbor_normals = gt_normals[inds]
            dotproduct = np.abs(np.sum(this_normals*neighbor_normals, axis=1))
            if np.any(dotproduct < ef1_dotproduct_threshold):
                flags[p] = True
    gt_edge_points = np.ascontiguousarray(gt_points[flags])

    #sample pred edge points
    indslist = pred_tree.query_radius(pred_points, ef1_radius)
    flags = np.zeros([len(pred_points)],bool)
    for p in range(len(pred_points)):
        inds = indslist[p]
        if len(inds)>0:
            this_normals = pred_normals[p:p+1]
            neighbor_normals = pred_normals[inds]
            dotproduct = np.abs(np.sum(this_normals*neighbor_normals, axis=1))
            if np.any(dotproduct < ef1_dotproduct_threshold):
                flags[p] = True
    pred_edge_points = np.ascontiguousarray(pred_points[flags])

    #ecd ef1

    if len(pred_edge_points)==0: pred_edge_points=np.zeros([486,3],np.float32)
    if len(gt_edge_points)==0:
        ecd1 = 0
        ecd2 = 0
        ef1 = 1
    else:
        # from gt to pred
        tree = KDTree(pred_edge_points)
        dist, inds = tree.query(gt_edge_points, k=1)
        erecall = np.sum(dist < ef1_threshold) / float(len(dist))
        gt2pred_mean_ecd1 = np.mean(dist)
        dist = np.square(dist)
        gt2pred_mean_ecd2 = np.mean(dist)

        # from pred to gt
        tree = KDTree(gt_edge_points)
        dist, inds = tree.query(pred_edge_points, k=1)
        eprecision = np.sum(dist < ef1_threshold) / float(len(dist))
        pred2gt_mean_ecd1 = np.mean(dist)
        dist = np.square(dist)
        pred2gt_mean_ecd2 = np.mean(dist)

        ecd1 = gt2pred_mean_ecd1+pred2gt_mean_ecd1
        ecd2 = gt2pred_mean_ecd2+pred2gt_mean_ecd2
        if erecall+eprecision > 0: ef1 = 2 * erecall * eprecision / (erecall + eprecision)
        else: ef1 = 0

    return cd1*100,cd2*(10**5),f1,nc,nr, ecd1*100,ecd2*10000,ef1

if __name__=='__main__':
    gt_fpath = opt.gt_path

    if 'abc' in gt_fpath.lower():
        gt_flist = [os.path.join(gt_fpath, file.strip()+'.ply') for file in open('../data/abc_test_files.txt')]
        test_size = len(gt_flist)
    else:
        gt_flist = glob(os.path.join(gt_fpath,'*.ply'))
        test_size = len(gt_flist)
        print(test_size)
    
    pred_fpath = opt.pred_path
    
    gt_path = [None]*test_size
    pred_surface_path = [None]*test_size
 
    for idx in range(test_size):
        gt_path[idx] = gt_flist[idx]
        pred_surface_path[idx] = os.path.join(pred_fpath, gt_path[idx].split('/')[-1])
         
    results = {'surface':[]}
    results_surface = np.zeros([test_size, 8], np.float32)
     
    with Pool(maxtasksperchild=1,processes=16) as pool:
        results['surface'] = pool.starmap(get_metrics, zip(gt_path, pred_surface_path))
    
    for i in range(test_size):
        results_surface[i] = np.array(results['surface'][i])

    print('\n==================================================================================')
    print("average results on all test samples:")

    print('surface:   (CD1,CD2,F1,NC,NR,ECD1,ECD2,EF1)=(%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f)'%tuple(np.mean(results_surface,axis=0)))
