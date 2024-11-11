import os
import argparse

import lovely_tensors as lt
import dust3r2colmap as dc

from pathlib import Path
from .src import interpolated_camera
from .src import parse_data

from scipy.spatial.transform import Rotation as R

# Specify Scene
scene_name = 'test'
split_keyword = 'DSC'

# Specify directories
image_dir = Path(f"./data/images/{scene_name}")
save_dir = Path(f"./data/scenes/{scene_name}")
save_dir.mkdir(exist_ok=True, parents=True)
folder_to_zip = f"./data/scenes/{scene_name}"
output_zip = f"./data/zips/{scene_name}_sparse.zip"

model_path = "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

# Hyperparameters
parser = argparse.ArgumentParser()
args = parser.parse_args()

args.batch_size = 1
args.schedule = 'cosine'
args.lr = 0.01
args.niter = 300

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


lt.monkey_patch()

if __name__ == '__main__':
    image_files = dc.import_train_images(scene_name, split_keyword, image_dir)
    scene = dc.train(model_path, image_files, args)
    dc.construct_colmap_dataset(scene, image_files, save_dir, split_keyword, extr='c2w')

    image_path = "" #여기는 더스터 출력인 images.txt path
    cameras_path =""

    camera_extrinsics = parse_data.parse_data_from_file("path", W2C = False)

    R1 = camera_extrinsics[8699]
    R2 = camera_extrinsics[8717]

    center = interpolated_camera.calculate_center_from_cameras(R1, R2)
    num = 90
    interpolated_matrix = interpolated_camera.imterpolated_matrices_with_center(R1, R2, center, num)

    interpolated_path = "" #새로운 images.txt 저장할 path

    images_file = os.path.join(interpolated_path ,'images.txt')
    # Generate images.txt content
    with open(images_file, 'w') as images_file:
        images_file.write("# Image list with two lines of data per image:\n")
        images_file.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        images_file.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for idx, matrix in enumerate(interpolated_matrix):
            img_number = idx
            # 회전 행렬을 쿼터니언으로 변환
            rotation_matrix = matrix[:3, :3]
            rotation = R.from_matrix(rotation_matrix)
            qw, qx, qy, qz = rotation.as_quat()
            tx, ty, tz = matrix[:3, 3]

            images_file.write(f"{img_number} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {img_number} {img_number}.png\nplaceholder\n")


    with open(cameras_path, "r") as fid:
        line = fid.readline()
        line = line.strip()
        if len(line) > 0 and line[0] != "#":
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
            width = int(elems[2])
            height = int(elems[3])
            params1 = float(elems[4])
            params2 = float(elems[5])
            params3 = float(elems[6])
            params4 = float(elems[7])

    cameras_file = os.path.join(interpolated_path ,'cameras.txt')

    with open(cameras_file, 'w') as cameras:
        HEADER = (
            "# Camera list with one line of data per camera:\n" +
            "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        )
        cameras.write(HEADER)
        for i in range (1,90):
            cameras_file.write(f"{i} {model} {width} {height} {params1} {params2} {params3} {params4}")



    # dc.construct_zip_files(folder_to_zip, output_zip)