import os
import bpy
import numpy as np
import mathutils

def load_colmap_data(cameras_path, images_path):
    # 카메라 파라미터 읽기
    cameras = {}
    with open(cameras_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[cam_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }

    # 이미지 및 외적 파라미터 읽기
    images = {}
    with open(images_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            if line.startswith('#') or not line.strip():
                continue
            if line == "placeholder":  # 'placeholder' 라인 건너뛰기
                continue
            parts = line.split()
            img_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            cam_id = int(parts[8])
            name = parts[9]
            images[img_id] = {
                'quaternion': [qw, qx, qy, qz],
                'translation': [tx, ty, tz],
                'camera_id': cam_id,
                'name': name
            }

    return cameras, images

def c2w_to_w2c(quaternion, translation):
    """Convert camera-to-world (c2w) to world-to-camera (w2c)."""
    c2w_rotation = mathutils.Quaternion(quaternion)
    c2w_translation = mathutils.Vector(translation)
    
    # w2c 변환
    w2c_rotation = c2w_rotation.conjugated()
    w2c_translation = -w2c_rotation @ c2w_translation
    return w2c_rotation, w2c_translation

def create_blender_cameras(cameras, images, display_size=0.05):
    for img_id, img_data in images.items():
        cam_data = cameras[img_data['camera_id']]

        # Blender 카메라 생성
        cam = bpy.data.cameras.new(f"Camera_{img_id}")
        cam.lens = cam_data['params'][0]  # Focal Length
        cam.sensor_width = cam_data['params'][2] * 2  # 센서 크기 추정

        cam_obj = bpy.data.objects.new(f"Camera_{img_id}", cam)
        bpy.context.collection.objects.link(cam_obj)

        cam_obj.scale = (display_size, display_size, display_size)
        # 위치 및 회전 설정 (c2w → w2c 변환 후 방향 보정)
        w2c_rotation, w2c_translation = c2w_to_w2c(
            img_data['quaternion'], img_data['translation']
        )
        cam_obj.location = w2c_translation
        cam_obj.rotation_mode = 'QUATERNION'
        cam_obj.rotation_quaternion = w2c_rotation


def create_image_planes(images, image_folder):
    """Creates planes with images over the corresponding cameras."""
    for img_id, img_data in images.items():
        img_name = img_data['name']
        img_path = os.path.join(image_folder, img_name)

        # Check if the image file exists
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        try:
            img = bpy.data.images.load(img_path)
        except RuntimeError as e:
            print(f"Failed to load image {img_name}: {e}")
            continue

        # Create a new material with the image texture
        mat = bpy.data.materials.new(name=f"Mat_{img_id}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        tex_image = nodes.new('ShaderNodeTexImage')
        tex_image.image = img
        links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

        # Create a plane for the image
        bpy.ops.mesh.primitive_plane_add(size=1)
        plane = bpy.context.object
        plane.name = f"ImagePlane_{img_id}"
        plane.data.materials.append(mat)

        # Position and rotate the plane to match the camera
        w2c_rotation, w2c_translation = c2w_to_w2c(
            img_data['quaternion'], img_data['translation']
        )
        plane.location = w2c_translation
        plane.rotation_mode = 'QUATERNION'
        plane.rotation_quaternion = w2c_rotation

        # Adjust plane size based on image aspect ratio
        aspect_ratio = img.size[0] / img.size[1]
        plane.scale = (aspect_ratio * 0.5, 0.5, 0.5)

        print(f"Successfully created plane for {img_name}.")


# 경로 설정
cameras_path = 'E:/ajhh9/Documents/Yai/Dust3r-to-COLMAP/data/interpolated_scenes/test/cameras.txt'  # 실제 경로 입력
images_path = 'E:/ajhh9/Documents/Yai/Dust3r-to-COLMAP/data/interpolated_scenes/test/images.txt'
image_folder = 'E:/ajhh9/Documents/Yai/Dust3r-to-COLMAP/data/images/test'  # 이미지 폴더 경로
# 데이터 로드 및 Blender 카메라 생성
cameras, images = load_colmap_data(cameras_path, images_path)
create_blender_cameras(cameras, images)
create_image_planes(images, image_folder)
