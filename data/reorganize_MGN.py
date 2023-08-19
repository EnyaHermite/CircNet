from glob import glob 
import os
import open3d as o3d
import numpy as np
import subprocess

# get the mesh files
data_dir = 'Multi-Garment_dataset'
folders = glob(data_dir+'/*')

mgn_file_list = []
for fold_path in folders:
    obj_files = glob(os.path.join(fold_path, '*.obj'))
    print(len(obj_files))

    for file in obj_files:
        if 'scan' in file or 'smpl_registered' in file:
            continue
        else:
            mgn_file_list.append(file.replace('Multi-Garment_dataset/',''))


# write the mesh files to new folder 'MGN'
new_data_dir = 'MGN'
if not os.path.exists(new_data_dir):
    os.makedirs(new_data_dir)

for mesh_path in mgn_file_list:
    mesh_path = 'Multi-Garment_dataset/' + mesh_path
    print(mesh_path)
    mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=False)
    strs = mesh_path.split('/')
    o3d.io.write_triangle_mesh(new_data_dir+'/%s_%s.ply'%(strs[-2],strs[-1][:-4]), mesh, 
                               write_ascii=True, write_triangle_uvs=False)


# delete the original folder
# Define the command as a list of strings
command = ["rm", "-rf", "Multi-Garment_dataset"]

# Use subprocess to run the command
try:
    subprocess.run(command, check=True, shell=False)
    print("Command executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Command failed with error code {e.returncode}: {e.stderr}")
except Exception as e:
    print(f"An error occurred: {e}")