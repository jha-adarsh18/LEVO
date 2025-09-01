import os
import yaml
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

class EventCalibrationExtractor(Dataset):
    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        self.calibrations = {}
        self.sequences = []
        self._discover_calibrations()

    def _discover_calibrations(self):
        for scenario in os.listdir(self.dataset_root):
            scenario_path = os.path.join(self.dataset_root, scenario)
            if not os.path.isdir(scenario_path):
                continue
            calibration_path = os.path.join(scenario_path, 'calibration')
            if os.path.exists(calibration_path):
                yaml_files = list(Path(calibration_path). glob('*.yaml'))
                if yaml_files:
                    yaml_file = yaml_files[0]
                    calib_data = self._load_yaml_calibration(yaml_file)
                    if calib_data:
                        slam_params = self._extract_slam_parameters(calib_data)
                        if slam_params:
                            self.calibrations[scenario] = slam_params
                            self.sequences.append({
                                'scenario': scenario,
                                'calibration_path': calibration_path,
                                'yaml_file': str(yaml_file)
                            })
    def _load_yaml_calibration(self, yaml_file):
        try:
            with open(yaml_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Error loading {yaml_file}: {e}")
            return None
    
    def _extract_slam_parameters(self, calib_data):
        if not calib_data or 'cam0' not in calib_data or 'cam1' not in calib_data:
            return None
        cam0 = calib_data['cam0']
        cam1 = calib_data['cam1']

        slam_params = {
            'left_camera': {
                'K': None,
                'D': cam0.get('distortion_coeffs', []),
                'distortion_model': cam0.get('distortion_model', 'equidistant'),
                'resolution': cam0.get('resolution', []),
                'T_cam_imu': cam0.get('T_cam_imu', []),
                'rectification_matrix': cam0.get('rectification_matrix', []),
                'projection_matrix': cam0.get('projection_matrix', [])
            },
            'right_camera': {
                'K': None,
                'D': cam1.get('distortion_coeffs', []),
                'distortion_model': cam1.get('distortion_model', 'equidistant'),
                'resolution': cam1.get('resolution', []),
                'T_cam_imu': cam1.get('T_cam_imu', []),
                'rectification_matrix': cam1.get('rectification_matrix', []),
                'projection_matrix': cam1.get('projection_matrix', [])
            },
            'stereo': {
                'T_right_left': cam1.get('T_cn_cnm1', []),
                'baseline': None,
                'R': None,
                't': None,
                'essential_matrix': None,
                'fundamental_matrix': None
            },
            'imu': {
                'T_left_imu': cam0.get('T_cam_imu', []),
                'T_right_imu': cam1.get('T_cam_imu', [])
            }
        }

        if cam0.get('intrinsics') and len(cam0['intrinsics']) >= 4:
            fx, fy, cx, cy = cam0['intrinsics'][:4]
            slam_params['left_camera']['K'] = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])

        if cam1.get('intrinsics') and len(cam1['intrinsics']) >= 4:
            fx, fy, cx, cy = cam1['intrinsics'][:4]
            slam_params['right_camera']['K'] = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])

        if cam1.get('T_cn_cnm1'):
            T_rl = np.array(cam1['T_cn_cnm1'])
            if T_rl.shape == (4,4):
                R = T_rl[:3, :3]
                t = T_rl[:3, 3]
                
                slam_params['stereo']['R'] = R
                slam_params['stereo']['t'] = t
                slam_params['stereo']['baseline'] = float(np.linalg.norm(t))

                if slam_params['left_camera']['K'] is not None and slam_params['right_camera']['K'] is not None:
                    t_skew = np.array([
                        [0, -t[2], t[1]],
                        [t[2], 0, -t[0]],
                        [-t[1], t[0], 0]
                    ])

                    E = t_skew @ R
                    slam_params['stereo']['essential_matrix'] = E
                    K_left = slam_params['left_camera']['K']
                    K_right = slam_params['right_camera']['K']
                    F = np.linalg.inv(K_right).T @ E @ np.linalg.inv(K_left)
                    slam_params['stereo']['fundamental_matrix'] = F

        
        return slam_params
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        scenario = seq_info['scenario']
        if scenario not in self.calibrations:
            return None
        calib = self.calibrations[scenario]

        K_left = calib['left_camera']['K']
        K_right = calib['right_camera']['K']
        R = calib['stereo']['R']
        t = calib['stereo']['t']
        P_left = None
        P_right = None

        if K_left is not None and K_right is not None and R is not None and t is not None:
            P_left = K_left @ np.hstack([np.eye(3), np.zeros((3,1))])
            P_right = K_right @ np.hstack([R, t.reshape(-1, 1)])

        return {
            'cameras': {
                'left': {
                    'K': K_left,
                    'D': np.array(calib['left_camera']['D']),
                    'resolution': calib['left_camera']['resolution'],
                    'distortion_model': calib['left_camera']['distortion_model'],
                },
                'right': {
                    'K': K_right,
                    'D': np.array(calib['right_camera']['D']),
                    'resolution': calib['right_camera']['resolution'],
                    'distortion_model': calib['right_camera']['distortion_model']
                }
            },
            'stereo': {
                'baseline': calib['stereo']['baseline'],
                'R': R,
                't': t,
                'essential_matrix': calib['stereo']['essential_matrix'],
                'fundamental_matrix': calib['stereo']['fundamental_matrix']
            },
            'triangulation': {
                'P_left': P_left,
                'P_right': P_right,
                'K_left': K_left,
                'K_right': K_right,
                'baseline': calib['stereo']['baseline']
            },
            'imu': {
                'T_left_imu': np.array(calib['imu']['T_left_imu']) if calib['imu']['T_left_imu'] else None,
                'T_right_imu': np.array(calib['imu']['T_right_imu']) if calib['imu']['T_right_imu'] else None
            },
            'rectification': {
                'left_rectification': calib['left_camera']['rectification_matrix'],
                'right_rectification': calib['right_camera']['rectification_matrix'],
                'left_projection': calib['left_camera']['projection_matrix'],
                'right_projection': calib['right_camera']['projection_matrix']
            },
            'sequence_info': {
                'scenario': scenario,
                'calibration_path': seq_info['calibration_path']
            }
        }
    
    def get_calibration_for_scenario(self, scenario):
        for idx, seq in enumerate(self.sequences):
            if seq['scenario'] == scenario:
                return self[idx]
        return
    
    def get_available_scenarios(self):
        return list(self.calibration.keys())
    
def extract_calibrations(dataset_root):
    return EventCalibrationExtractor(dataset_root)

if __name__ == "__main__":
    dataset_root = r"/home/adarsh/Documents/SRM/dataset/train" #replace with actual path if required
    dataset = extract_calibrations(dataset_root)
    print(f"found {len(dataset)} calibrations")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"sample from {sample['sequence_info']['scenario']}:")
        print(f"left camera K shape: {sample['cameras']['left']['K'].shape}")
        print(f"right camera K shape: {sample['cameras']['right']['K'].shape}")
        print(f"baseline: {sample['stereo']['baseline']:.6f}m")
        print(f"distortion model: {sample['cameras']['left']['distortion_model']}")
        print(f"resolution: {sample['cameras']['left']['resolution']}")
        
        has_K = sample['cameras']['left']['K'] is not None and sample['cameras']['right']['K'] is not None
        has_stereo = sample['stereo']['R'] is not None and sample['stereo']['t'] is not None
        has_imu = sample['imu']['T_left_imu'] is not None
        has_triangulation = sample['triangulation']['P_left'] is not None

        print(f"camera matrices: {'available' if has_K else 'unavailable'}")
        print(f"stereo geometry: {'available' if has_stereo else 'unavilable'}")
        print(f"imu integration: {'available' if has_imu else 'unavailable'}")
        print(f"triangulation ready: {'availalble' if has_triangulation else 'unavailable'}")
         