import numpy as np
import open3d as o3d
import os
from glob import glob
from src.postprocess import PostProcess
from functools import partial
from multiprocessing import Pool, freeze_support
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--read_path', required=True, help='directory to the primitive meshes')
opt = parser.parse_args()


def compute_angles(points, triangles):
    A = points[triangles[:, 0], :]  # triangle vertex A
    B = points[triangles[:, 1], :]  # triangle vertex B
    C = points[triangles[:, 2], :]  # triangle vertex C

    # calculate the angles
    l_AB = np.sqrt(np.sum((A - B) ** 2, axis=-1))
    l_AC = np.sqrt(np.sum((A - C) ** 2, axis=-1))
    l_BC = np.sqrt(np.sum((B - C) ** 2, axis=-1))
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
    return angles

def check_manifold(triangles):
    v1 = triangles[:,0]
    v2 = triangles[:,1]
    v3 = triangles[:,2]
    edges = np.reshape(np.stack([v1, v2, v2, v3, v1, v3], axis=1), [-1, 2])
    uni_edges, uni_cnts = np.unique(edges, return_counts=True, axis=0)
    uni_degrees, degree_cnts = np.unique(uni_cnts, return_counts=True, axis=0)
    print("edge degree counts: ", uni_degrees, degree_cnts)
    # assert(np.max(uni_degrees)<=2)
    return

def post_process_mesh(readpath, writepath, fill_holes=True):
    PPS = PostProcess()
    mesh = o3d.io.read_triangle_mesh(readpath)
    vertices = np.asarray(mesh.vertices).astype('float32')
    triangles = np.asarray(mesh.triangles).astype('int32')
    triangles = PPS.clean_mesh(vertices, triangles, fill_holes=fill_holes)

    new_mesh = mesh
    new_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    o3d.io.write_triangle_mesh(writepath,new_mesh,write_ascii=True)
    return

def main(read_path, write_path, test_size):
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    if 'manifold' in write_path:
        fill_holes = False
    elif 'surface' in write_path:
        fill_holes = True
    else:
        raise Exception("fill_holes not defined!!")

    read_files = [None]*test_size
    write_files = [None]*test_size
    
    read_files = glob(os.path.join(read_path,'*.ply'))
    for i in range(test_size):
        write_files[i] = os.path.join(write_path, read_files[i].split('/')[-1])

    with Pool() as pool:
        pool.starmap(partial(post_process_mesh, fill_holes=fill_holes), zip(read_files, write_files))
    return



if __name__=="__main__":

    write_path = opt.read_path.replace('primitive','manifold')
    test_size = len(glob(os.path.join(opt.read_path,'*.ply')))
    print(test_size)
    freeze_support()
    main(opt.read_path, write_path, test_size)
