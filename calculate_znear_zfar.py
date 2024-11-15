import numpy as np
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData

def parse_cameras(cameras_file):
    with open(cameras_file, 'r') as f:
        lines = f.readlines()
    
    cameras = {}
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        camera_id = int(parts[0])
        width = int(parts[2])
        height = int(parts[3])
        fx = float(parts[4])
        fy = float(parts[5])
        cx = float(parts[6])
        cy = float(parts[7])
        cameras[camera_id] = {'width': width, 'height': height, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    return cameras

def parse_images(images_file):
    with open(images_file, 'r') as f:
        lines = f.readlines()
    
    poses = {}
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith('#') or not line.strip():
            continue
        if line == "placeholder":  # 'placeholder' 라인 건너뛰기
            continue
        
        parts = lines[i].split()
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
        translation = np.array([tx, ty, tz]).reshape(3, 1)
        pose = np.hstack((rotation, translation))  # c2w pose
        bottom_row = np.array([[0, 0, 0, 1]])  # Homogeneous coordinates
        pose = np.vstack((pose, bottom_row))
        poses[image_id] = pose
    return poses

def parse_points3d(points3d_file):
    plydata = PlyData.read(points3d_file)
    points = {}
    for i, vertex in enumerate(plydata['vertex']):
        points[i] = np.array([vertex['x'], vertex['y'], vertex['z']])
    return points

def calculate_znear_zfar(poses, pts3d):
    zvals = []
    
    for pose in poses.values():
        camera_position = pose[:3, 3]  # Translation vector in c2w
        for pt_id, pt in pts3d.items():
            point_position = pt
            zval = np.dot(pose[:3, 2], (point_position - camera_position))  # Depth along Z-axis
            zvals.append(zval)
    
    zvals = np.array(zvals)
    valid_z = zvals[zvals > 0]  # Consider only valid (positive) depth values
    znear = np.percentile(valid_z, 0.1)
    zfar = np.percentile(valid_z, 99.9)
    
    return znear, zfar

# Load and calculate znear, zfar
cameras_path = './data/scenes/book_cap/time_step_1/sparse/0/cameras.txt'
images_path = './data/scenes/book_cap/time_step_1/sparse/0/images.txt'
points3d_path = './data/scenes/book_cap/time_step_1/sparse/0/points3D.ply'

cameras = parse_cameras(cameras_path)
poses = parse_images(images_path)
pts3d = parse_points3d(points3d_path)

znear, zfar = calculate_znear_zfar(poses, pts3d)

print(f"ZNear: {znear}, ZFar: {zfar}")
