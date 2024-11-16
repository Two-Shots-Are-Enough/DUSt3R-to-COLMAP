import sys
sys.path.append("./dust3r")
sys.path.append("./gaussian-splatting")

import os
import argparse
import cv2  # Assuming OpenCV is used for image saving
import torch
import trimesh
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import lovely_tensors as lt

from PIL import Image
from pathlib import Path
from typing import NamedTuple, Optional

from utils import BasicPointCloud, inv, rotmat2qvec, focal2fov, fov2focal
# from scene.dataset_readers import storePly

from dust3r.inference import inference
from dust3r.model import load_model
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: np.ndarray
    FovX: np.ndarray
    image: np.ndarray
    image_path: str
    image_name: str
    width: int
    height: int
    mask: Optional[np.ndarray] = None
    mono_depth: Optional[np.ndarray] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    render_cameras: Optional[list[CameraInfo]] = None


def import_train_images(scene_name, split_keyword, image_dir):
    Path.ls = lambda x: list(x.iterdir())

    image_files = [str(x) for x in image_dir.ls() if x.suffix.lower() in ['.png', '.jpg']]
    image_files = sorted(image_files, key=lambda x: int(x.split(split_keyword)[-1].split('.')[0])) 

    return image_files


def train(model_path, image_files, args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)

    images = load_images(image_files, size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

    output = inference(pairs, model, device, batch_size=args.batch_size)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=args.niter, schedule=args.schedule, lr=args.lr)
    print(loss)
    return scene


def init_filestructure(save_path):
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    images_path = save_path / 'images'
    masks_path = save_path / 'masks'
    sparse_path = save_path / 'sparse/0'

    images_path.mkdir(exist_ok=True, parents=True)
    masks_path.mkdir(exist_ok=True, parents=True)
    sparse_path.mkdir(exist_ok=True, parents=True)

    return save_path, images_path, masks_path, sparse_path


def save_images_masks(split_keyword, imgs, masks, image_files, images_path, masks_path):
    # Saving images and optionally masks/depth maps
    for img_path, image, mask in zip(image_files, imgs, masks):
        # DSC 뒤 숫자 추출
        img_number = int(img_path.split(split_keyword)[-1].split('.')[0])

        # 저장 경로 설정
        image_save_path = images_path / f"{img_number}.png"
        mask_save_path = masks_path / f"{img_number}.png"

        original_image = cv2.imread(img_path)
        cv2.imwrite(str(image_save_path), original_image)
        # image[~mask] = 1.
        # rgb_image = cv2.cvtColor(image*255, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(str(image_save_path), rgb_image)

        mask = np.repeat(np.expand_dims(mask, -1), 3, axis=2)*255
        Image.fromarray(mask.astype(np.uint8)).save(mask_save_path)


def save_cameras(split_keyword, focals, principal_points, image_files, sparse_path, imgs_shape):
    # Save cameras.txt
    cameras_file = Path(sparse_path)/'cameras.txt'
    with open(cameras_file, 'w') as cameras_file:
        cameras_file.write("# Camera list with one line of data per camera:\n")
        cameras_file.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for img_path, focal, pp in zip(image_files, focals, principal_points):
            # DSC 뒤 숫자 추출
            img_number = int(img_path.split(split_keyword)[-1].split('.')[0])
            cameras_file.write(f"{img_number} PINHOLE {imgs_shape[2]} {imgs_shape[1]} "
                    f"{focal[0]} {focal[0]} {pp[0]} {pp[1]}\n")


def save_images_txt(split_keyword, extrinsics, image_files, sparse_path):
     # Save images.txt
    images_file = Path(sparse_path) / 'images.txt'
    # Generate images.txt content
    with open(images_file, 'w') as images_file:
        images_file.write("# Image list with two lines of data per image:\n")
        images_file.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        images_file.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, (img_path, pose) in enumerate(zip(image_files, extrinsics)):
            img_number = int(img_path.split(split_keyword)[-1].split('.')[0])

            # 회전 행렬을 쿼터니언으로 변환
            rotation_matrix = pose[:3, :3]
            qw, qx, qy, qz = rotmat2qvec(rotation_matrix)
            tx, ty, tz = pose[:3, 3]

            images_file.write(f"{img_number} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {img_number} {img_number}.png\n\n")


def save_pointcloud_with_normals(imgs, pts3d, msk, sparse_path):
    pc = get_pc(imgs, pts3d, msk)  # Assuming get_pc is defined elsewhere and returns a trimesh point cloud

    # Define a default normal, e.g., [0, 1, 0]
    default_normal = [0, 1, 0]

    # Prepare vertices, colors, and normals for saving
    vertices = pc.vertices
    colors = pc.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))

    save_path = Path(sparse_path) / 'points3D.ply'

    # Construct the header of the PLY file
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(len(vertices))

    # Write the PLY file
    with open(save_path, 'w') as ply_file:
        ply_file.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            ply_file.write('{} {} {} {} {} {} {} {} {}\n'.format(
                vertex[0], vertex[1], vertex[2],
                int(color[0]), int(color[1]), int(color[2]),
                normal[0], normal[1], normal[2]
            ))


def get_pc(imgs, pts3d, mask):
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)
    mask = to_numpy(mask)

    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)])

    pts = pts.reshape(-1, 3)[::3]
    col = col.reshape(-1, 3)[::3]

    #mock normals:
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))

    pct = trimesh.PointCloud(pts, colors=col)
    pct.vertices_normal = normals  # Manually add normals to the point cloud

    return pct#, pts


def save_pointcloud(imgs, pts3d, msk, sparse_path):
    save_path = Path(sparse_path) / 'points3D.ply'
    pc = get_pc(imgs, pts3d, msk)

    pc.export(save_path)


def construct_colmap_dataset(scene, image_files, save_dir, split_keyword, extr='c2w'): # extrinsic: 'c2w' or 'w2c' 
    intrinsics = scene.get_intrinsics().detach().cpu().numpy()

    if extr == 'w2c':
        extrinsics = inv(scene.get_im_poses().detach()).cpu().numpy()
    if extr == 'c2w':
        extrinsics = scene.get_im_poses().detach().cpu().numpy()

    principal_points = scene.get_principal_points().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()
    imgs = np.array(scene.imgs)
    pts3d = [i.detach() for i in scene.get_pts3d()]
    depth_maps = [i.detach() for i in scene.get_depthmaps()]

    min_conf_thr = 20
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    masks = to_numpy(scene.get_masks())

    save_path, images_path, masks_path, sparse_path = init_filestructure(save_dir)
    save_pointcloud_with_normals(imgs, pts3d, masks, sparse_path)
    save_images_masks(split_keyword, imgs, masks, image_files, images_path, masks_path)
    save_cameras(split_keyword, focals, principal_points, image_files, sparse_path, imgs_shape=imgs.shape)
    save_images_txt(split_keyword, extrinsics, image_files, sparse_path)


def construct_zip_files(scene_path, output_path):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # sparse path
    sparse_folder_path = os.path.join(scene_path, 'sparse')

    # zip generate
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(sparse_folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(scene_path))
                zipf.write(file_path, arcname)