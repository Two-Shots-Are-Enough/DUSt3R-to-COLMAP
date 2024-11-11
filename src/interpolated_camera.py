import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
#방향 가운데로 맞춰짐.

def slerp(q1, q2, t):
    """
    Perform Spherical Linear Interpolation (SLERP) between two quaternions q1 and q2 at parameter t.
    """
    # Convert quaternions to unit quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute the dot product
    dot = np.dot(q1, q2)

    # If the dot product is negative, negate one quaternion to take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If the dot product is close to 1, use linear interpolation (LERP)
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        result = result / np.linalg.norm(result)
        return result
    
    # Otherwise, compute the SLERP
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    q3 = q2 - q1 * dot
    q3 = q3 / np.linalg.norm(q3)
    
    # Compute the final interpolated quaternion
    q_interpolated = q1 * np.cos(theta) + q3 * np.sin(theta)
    return q_interpolated


def calculate_center_from_cameras(extrinsic_matrix1, extrinsic_matrix2):
    """
    두 카메라의 extrinsic 행렬을 통해 중심점을 계산합니다.
    
    :param extrinsic_matrix1: 첫 번째 카메라의 4x4 extrinsic 행렬
    :param extrinsic_matrix2: 두 번째 카메라의 4x4 extrinsic 행렬
    :return: 두 카메라가 바라보는 중심점 (3D 좌표)
    """
    # 각 카메라 위치 추출
    cam_pos1 = extrinsic_matrix1[:3, 3]
    cam_pos2 = extrinsic_matrix2[:3, 3]
    
    # 각 카메라의 z축 방향 벡터 (시선 방향) 추출
    direction1 = extrinsic_matrix1[:3, 2]  # z-axis
    direction2 = extrinsic_matrix2[:3, 2]  # z-axis
    
    # 시선 방향 벡터 정규화
    direction1 = direction1 / np.linalg.norm(direction1)
    direction2 = direction2 / np.linalg.norm(direction2)
    
    # 두 카메라가 바라보는 중심점 계산 (직선의 가장 가까운 점)
    w0 = cam_pos1 - cam_pos2
    a = np.dot(direction1, direction1)
    b = np.dot(direction1, direction2)
    c = np.dot(direction2, direction2)
    d = np.dot(direction1, w0)
    e = np.dot(direction2, w0)
    
    denominator = a * c - b * b
    if abs(denominator) < 1e-6:
        # 두 시선이 평행한 경우 중심을 단순히 두 위치의 중간점으로 설정
        center = (cam_pos1 + cam_pos2) / 2
    else:
        s = (b * e - c * d) / denominator
        t = (a * e - b * d) / denominator
        closest_point1 = cam_pos1 + s * direction1
        closest_point2 = cam_pos2 + t * direction2
        # 가장 가까운 점의 중점 계산
        center = (closest_point1 + closest_point2) / 2
    
    return center


def interpolate_matrices_with_center(extrinsic_matrix1, extrinsic_matrix2, center, num_interpolations):
    """
    Interpolate between two 4x4 extrinsic matrices using a shared center point.
    
    :param extrinsic_matrix1: First 4x4 extrinsic matrix
    :param extrinsic_matrix2: Second 4x4 extrinsic matrix
    :param center: 3D point to act as the center of the spherical interpolation
    :param num_interpolations: Number of interpolated matrices to generate
    :return: List of interpolated extrinsic matrices
    """
    # Extract camera positions and rotation matrices
    cam_pos1 = extrinsic_matrix1[:3, 3]
    cam_pos2 = extrinsic_matrix2[:3, 3]
    R1 = extrinsic_matrix1[:3, :3]
    R2 = extrinsic_matrix2[:3, :3]
    
    # Calculate quaternions from rotation matrices
    q1 = R.from_matrix(R1).as_quat()
    q2 = R.from_matrix(R2).as_quat()
    
    # Initialize list to hold interpolated matrices
    interpolated_matrices = []
    
    for i in range(num_interpolations):
        alpha = i / (num_interpolations - 1)
        
        # Interpolate position on the spherical surface
        interp_pos = (1 - alpha) * cam_pos1 + alpha * cam_pos2
        direction = (center - interp_pos) / np.linalg.norm(center - interp_pos)
        
        # Interpolate rotation using SLERP
        q_interpolated = slerp(q1, q2, alpha)
        
        # Calculate interpolated rotation matrix
        interpolated_R = R.from_quat(q_interpolated).as_matrix()
        
        # Create new extrinsic matrix
        interpolated_matrix = np.eye(4)
        interpolated_matrix[:3, :3] = interpolated_R
        interpolated_matrix[:3, 3] = interp_pos
        
        interpolated_matrices.append(interpolated_matrix)
    
    return interpolated_matrices
