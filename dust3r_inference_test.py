import os
import argparse
import lovely_tensors as lt
import dust3r2colmap as dc
import numpy as np
from pathlib import Path
from src import interpolated_camera
from src import parse_data
from scipy.spatial.transform import Rotation as R


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


# Specify Scene Info
scene_name = 'test'
split_keyword = 'DSC'

# Specify directories: DUST3R to COLMAP
image_dir = Path(f"./data/images/{scene_name}")
save_dir = Path(f"./data/scenes/{scene_name}")
save_dir.mkdir(exist_ok=True, parents=True)
folder_to_zip = f"./data/scenes/{scene_name}"
output_zip = f"./data/zips/{scene_name}_sparse.zip"

model_path = "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

# Specify directories: Camera Interpolation
image_path = os.path.join(save_dir, 'sparse/0', 'images.txt') #여기는 더스터 출력인 images.txt path
cameras_path = os.path.join(save_dir, 'sparse/0', 'cameras.txt')
interpolated_path = Path(f"./data/interpolated_scenes/{scene_name}/") #새로운 images.txt 저장할 path
images_file = os.path.join(interpolated_path ,'images.txt') # interpolated 포함 images.txt
cameras_file = os.path.join(interpolated_path ,'cameras.txt') # interpolated 포함 cameras.txt 


# Hyperparameters
parser = argparse.ArgumentParser()
args = parser.parse_args()

args.batch_size = 1
args.schedule = 'cosine'
args.lr = 0.01
args.niter = 300

num = 90 # total number of cam

# ===== Modify scene.gaussian_model ===== #

file_path = os.path.join("gaussian-splatting", "scene", "gaussian_model.py")

old_import = "from simple_knn._C import distCUDA2"

new_import = """\nfrom scipy.spatial import KDTree
import torch

def distCUDA2(points):
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)

    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)
"""

with open(file_path, 'r') as file:
    file_data = file.read()

if old_import in file_data:
    file_data = file_data.replace(old_import, new_import)

    with open(file_path, 'w') as file:
        file.write(file_data)

    print(f"'{file_path}' 파일에서 '{old_import}'이(가) 성공적으로 대체되었습니다.")
else:
    print(f"'{file_path}' 파일에서 '{old_import}' 문구를 찾을 수 없습니다.")

# ==================== #


lt.monkey_patch()

if __name__ == '__main__':
    image_files = dc.import_train_images(scene_name, split_keyword, image_dir)
    scene = dc.train(model_path, image_files, args)
    dc.construct_colmap_dataset(scene, image_files, save_dir, split_keyword, extr='c2w')
    # dc.construct_zip_files(folder_to_zip, output_zip)

    camera_extrinsics = parse_data.parse_data_from_file(image_path, W2C = False)

    # Specify Interpolation Info
    R1 = camera_extrinsics[8699]
    R2 = camera_extrinsics[8778]

    center = interpolated_camera.calculate_center_from_cameras(R1, R2)
    interpolated_matrix = interpolated_camera.interpolate_matrices_with_center(R1, R2, center, num)

    # Generate images.txt content
    with open(images_file, 'w') as images_file:
        images_file.write("# Image list with two lines of data per image:\n")
        images_file.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        images_file.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for idx, matrix in enumerate(interpolated_matrix):
            img_number = idx+1
            # 회전 행렬을 쿼터니언으로 변환
            rotation_matrix = matrix[:3, :3]
            rotation = rotmat2qvec(rotation_matrix)
            qw, qx, qy, qz = rotation
            tx, ty, tz = matrix[:3, 3]
            images_file.write(f"{img_number} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {img_number} {img_number}.png\nplaceholder\n")
            
    # Read the original cameras.txt file
    with open(cameras_path, 'r') as file:
        lines = file.readlines()

    # Extract camera information for C1 and C2
    camera_info = {}
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.strip().split()
        camera_id = int(parts[0])
        camera_info[camera_id] = parts[1:]  # Store model, width, height, and params

    # Get camera info for r1 and r2
    info_C1 = camera_info[8699]
    info_C2 = camera_info[8778]

    # Create new cameras.txt with interpolated IDs and r1's params
    with open(cameras_file, 'w') as new_file:
        new_file.write("# Camera list with one line of data per camera:\n")
        new_file.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")

        # Generate 88 images for r1 + 2 boundary images
        for i in range(1, 91):
            if i <= 89:
                new_file.write(f"{i} {' '.join(info_C1)}\n")
            else:
                new_file.write(f"{i} {' '.join(info_C2)}\n")