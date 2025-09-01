import numpy as np
from scipy.spatial.transform import Rotation as R

def load_mocap_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) == 8:
                    timestamp = float(parts[0])
                    pose = [float(x) for x in parts[1:]]
                    data.append([timestamp] + pose)
    return np.array(data)

def transform_pose_to_camera_frame(mocap_pose, extrinsics):
    t_mocap = np.array(mocap_pose[:3])
    q_mocap = mocap_pose[3:]
    
    t_cam_to_mocap = np.array([extrinsics['px'], extrinsics['py'], extrinsics['pz']])
    q_cam_to_mocap = np.array([extrinsics['qx'], extrinsics['qy'], extrinsics['qz'], extrinsics['qw']])
    
    R_mocap = R.from_quat(q_mocap)
    R_cam_to_mocap = R.from_quat(q_cam_to_mocap)
    
    R_mocap_to_cam = R_cam_to_mocap.inv()
    t_mocap_to_cam = -R_mocap_to_cam.apply(t_cam_to_mocap)
    
    t_camera = R_mocap_to_cam.apply(t_mocap - t_cam_to_mocap)
    R_camera = R_mocap_to_cam * R_mocap
    q_camera = R_camera.as_quat()
    
    return np.concatenate([t_camera, q_camera])

def convert_mocap_to_camera_frame(input_file, output_file, extrinsics):
    data = load_mocap_data(input_file)
    
    converted_poses = []
    
    for row in data:
        timestamp_us = int(row[0])  # Convert seconds → microseconds, keep int64
        mocap_pose = row[1:]
        camera_pose = transform_pose_to_camera_frame(mocap_pose, extrinsics)
        converted_poses.append([timestamp_us] + camera_pose.tolist())
    
    with open(output_file, 'w') as f:
        for pose in converted_poses:
            ts = np.int64(pose[0])
            tx, ty, tz = pose[1:4]
            qx, qy, qz, qw = pose[4:8]
            f.write(f"{ts} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    
    print(f"✅ Converted {len(converted_poses)} poses to camera frame with int64 microsecond timestamps")

# Main execution
if __name__ == "__main__":
    extrinsics = {
        "px": -0.034803406191293906,
        "py": 0.05971773350374604,
        "pz": -0.03694727254557562,
        "qx": -0.7015384510188221,
        "qy": 0.7125976574153885,
        "qz": 0.006516107167940321,
        "qw": 0.002433256169906342
    }
    
    input_file = "/media/adarsh/One Touch/EventSLAM/TUM-VIE/mocap-desk-vi_gt_data/mocap_data.txt"
    output_file = "/media/adarsh/One Touch/EventSLAM/dataset/train/mocap_desk/pose.txt"
    
    convert_mocap_to_camera_frame(input_file, output_file, extrinsics)
