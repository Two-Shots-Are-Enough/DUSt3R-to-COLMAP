import numpy as np

def W2C_C2W_transform(matrix):
    W2C_matrix = matrix
    C2W_matrix = np.linarg.inv(W2C_matrix)

    return C2W_matrix

def quaternion_to_matrix(w, x, y, z):
    # 4x4 회전 행렬 초기화
    w, x, y, z = w, x, y, z
    R = np.zeros((4, 4))

    # 회전 행렬 요소 계산
    R[0, 0] = 1 - 2 * (y**2 + z**2)
    R[0, 1] = 2 * (x * y - w * z)
    R[0, 2] = 2 * (x * z + w * y)
    R[1, 0] = 2 * (x * y + w * z)
    R[1, 1] = 1 - 2 * (x**2 + z**2)
    R[1, 2] = 2 * (y * z - w * x)
    R[2, 0] = 2 * (x * z - w * y)
    R[2, 1] = 2 * (y * z + w * x)
    R[2, 2] = 1 - 2 * (x**2 + y**2)

    # 마지막 행과 열을 단위로 설정
    R[3, 3] = 1

    return R


def parse_data_from_file(filename, W2C = True):
    #dust3r로 뽑은 colmap iamges.txt를 읽어오는 함수
  
    camera_extrinsics = {}
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i in range(3, len(lines), 2):  # Skip every second line (placeholder)
            parts = lines[i].strip().split()
            if len(parts) < 10:
                continue
            
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = parts
            
            # Convert values to appropriate types
            qw, qx, qy, qz = map(float, [qw, qx, qy, qz])
            tx, ty, tz = map(float, [tx, ty, tz])
            camera_id = int(camera_id)
            
            # Create extrinsic matrix
            extrinsic_matrix = np.eye(4)
            extrinsic_matrix = quaternion_to_matrix(qw, qx, qy, qz)
            extrinsic_matrix[:3, 3] = [tx, ty, tz]  # Set translation vector
            
            if W2C:
                extrinsic_matrix = W2C_C2W_transform(extrinsic_matrix)
            
            # Store the extrinsic matrix in the dictionary
            camera_extrinsics[camera_id] = extrinsic_matrix
    
    return camera_extrinsics
