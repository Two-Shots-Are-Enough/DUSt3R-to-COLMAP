import os
import argparse

import lovely_tensors as lt
import dust3r2colmap as dc

from pathlib import Path


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
    dc.construct_colmap_dataset(scene, image_files, save_dir, split_keyword, extr='w2c')
    # dc.construct_zip_files(folder_to_zip, output_zip)
