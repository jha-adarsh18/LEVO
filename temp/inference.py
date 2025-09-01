import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import os

# Import components
from loadeventdata import extract_events
from pointnet_events import EventPointNet
from neural_ode_module import EndToEndEventSLAM

class EventSLAMInference:
    """
    Inference wrapper for the trained Event SLAM model
    """
    def __init__(self, model_path: str, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Load model
        self.model, self.config = self._load_model(model_path)
        self.model.eval()
        
        print(f"Event SLAM model loaded on {self.device}")
        print(f"Latent dimension: {self.config['latent_dim']}")
    
    def _load_model(self, model_path: str) -> Tuple[EndToEndEventSLAM, dict]:
        """Load the trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        # Recreate model architecture
        pointnet = EventPointNet(latent_dim=config['latent_dim'], use_stereo=True)
        model = EndToEndEventSLAM(
            pointnet_model=pointnet,
            latent_dim=config['latent_dim'],
            ode_hidden_dim=config['ode_hidden_dim']
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, config
    
    def process_event_sequence(self, left_events_seq: torch.Tensor, 
                             right_events_seq: torch.Tensor,
                             time_deltas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a sequence of event data and return poses and latent trajectory
        Args:
            left_events_seq: [seq_len, max_events, 4] left camera events
            right_events_seq: [seq_len, max_events, 4] right camera events  
            time_deltas: [seq_len] time intervals between frames
        Returns:
            poses: [seq_len, 7] predicted poses (translation + quaternion)
            latent_trajectory: [seq_len, latent_dim] latent space trajectory
        """
        with torch.no_grad():
            # Add batch dimension
            left_batch = left_events_seq.unsqueeze(0).to(self.device)  # [1, seq_len, max_events, 4]
            right_batch = right_events_seq.unsqueeze(0).to(self.device)  # [1, seq_len, max_events, 4]
            time_deltas = time_deltas.to(self.device)
            
            # Forward pass
            pred_poses, latent_trajectory = self.model(left_batch, right_batch, time_deltas)
            
            # Remove batch dimension
            poses = pred_poses.squeeze(0).cpu()  # [seq_len, 7]
            latent = latent_trajectory.squeeze(0).cpu()  # [seq_len, latent_dim]
            
            return poses, latent
    
    def predict_future_trajectory(self, left_events_seq: torch.Tensor,
                                right_events_seq: torch.Tensor,
                                time_deltas: torch.Tensor,
                                future_horizon: float = 50e-6) -> torch.Tensor:
        """
        Predict future poses given a history of events
        Args:
            left_events_seq: [seq_len, max_events, 4] history of left events
            right_events_seq: [seq_len, max_events, 4] history of right events
            time_deltas: [seq_len] historical time intervals
            future_horizon: future time horizon in seconds
        Returns:
            future_poses: [num_future, 7] predicted future poses
        """
        with torch.no_grad():
            # Create future time points (assuming 5μs intervals)
            num_future = int(future_horizon / 5e-6)
            future_times = torch.arange(1, num_future + 1).float() * 5e-6
            future_times = future_times.to(self.device)
            
            # Add batch dimension
            left_batch = left_events_seq.unsqueeze(0).to(self.device)
            right_batch = right_events_seq.unsqueeze(0).to(self.device)
            time_deltas = time_deltas.to(self.device)
            
            # Predict future
            future_poses = self.model.predict_future_poses(
                left_batch, right_batch, time_deltas, future_times
            )
            
            return future_poses.squeeze(0).cpu()  # [num_future, 7]
    
    def process_dataset_sequence(self, dataset_root: str, sequence_idx: int = 0,
                               sequence_length: int = 10) -> dict:
        """
        Process a sequence from the dataset
        Args:
            dataset_root: path to dataset
            sequence_idx: which sequence to process
            sequence_length: number of frames to process
        Returns:
            results dictionary with poses, latent features, and metadata
        """
        # Load dataset
        dataset = extract_events(dataset_root, duration_us=5, max_events=2048)
        
        if sequence_idx >= len(dataset):
            raise ValueError(f"Sequence index {sequence_idx} out of range (max: {len(dataset)-1})")
        
        # Collect sequence data
        left_events_seq = []
        right_events_seq = []
        true_poses = []
        timestamps = []
        
        for i in range(min(sequence_length, len(dataset) - sequence_idx)):
            sample = dataset.get_pointnet_sample(sequence_idx + i)
            
            left_events_seq.append(sample['left_tensor'])
            right_events_seq.append(sample['right_tensor'])
            true_poses.append(sample['pose'])
            timestamps.append(sample['metadata']['t_start'])
        
        # Convert to tensors
        left_events_seq = torch.stack(left_events_seq)  # [seq_len, max_events, 4]
        right_events_seq = torch.stack(right_events_seq)  # [seq_len, max_events, 4]
        true_poses = torch.stack(true_poses)  # [seq_len, 7]
        timestamps = torch.tensor(timestamps)  # [seq_len]
        
        # Compute time deltas
        time_deltas = torch.zeros_like(timestamps)
        time_deltas[1:] = timestamps[1:] - timestamps[:-1]
        
        # Process with model
        pred_poses, latent_trajectory = self.process_event_sequence(
            left_events_seq, right_events_seq, time_deltas
        )
        
        return {
            'predicted_poses': pred_poses.numpy(),
            'true_poses': true_poses.numpy(),
            'latent_trajectory': latent_trajectory.numpy(),
            'timestamps': timestamps.numpy(),
            'sequence_info': {
                'start_idx': sequence_idx,
                'length': len(left_events_seq)
            }
        }

def visualize_results(results: dict, save_path: str = None):
    """
    Visualize the SLAM results
    """
    pred_poses = results['predicted_poses']
    true_poses = results['true_poses']
    latent_trajectory = results['latent_trajectory']
    timestamps = results['timestamps']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 3D Trajectory
    ax1 = axes[0, 0]
    ax1.plot(true_poses[:, 0], true_poses[:, 1], 'b-o', label='Ground Truth', markersize=4)
    ax1.plot(pred_poses[:, 0], pred_poses[:, 1], 'r-s', label='Predicted', markersize=4)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('2D Trajectory Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Position Error over Time
    ax2 = axes[0, 1]
    position_error = np.linalg.norm(pred_poses[:, :3] - true_poses[:, :3], axis=1)
    ax2.plot(timestamps * 1000, position_error, 'g-', linewidth=2)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Position Error')
    ax2.set_title('Position Error Over Time')
    ax2.grid(True)
    
    # 3. Latent Space Dynamics (first 3 dimensions)
    ax3 = axes[1, 0]
    for i in range(min(3, latent_trajectory.shape[1])):
        ax3.plot(timestamps * 1000, latent_trajectory[:, i], label=f'Dim {i+1}')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Latent Feature Value')
    ax3.set_title('Latent Space Dynamics (First 3 Dims)')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Translation Components
    ax4 = axes[1, 1]
    ax4.plot(timestamps * 1000, true_poses[:, 0], 'b-', label='True X')
    ax4.plot(timestamps * 1000, pred_poses[:, 0], 'r--', label='Pred X')
    ax4.plot(timestamps * 1000, true_poses[:, 1], 'g-', label='True Y')
    ax4.plot(timestamps * 1000, pred_poses[:, 1], 'm--', label='Pred Y')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Position')
    ax4.set_title('Translation Components')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to {save_path}")
    
    plt.show()

def main():
    """
    Example usage of the inference system
    """
    model_path = "event_slam_end2end.pth"
    dataset_root = "/home/adarsh/Documents/SRM/dataset/train"
    
    print("Event SLAM Inference Demo")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        print("Please train the model first using the training script.")
        return
    
    # Initialize inference system
    slam_inference = EventSLAMInference(model_path)
    
    # Process a sequence from the dataset
    print("Processing dataset sequence...")
    try:
        results = slam_inference.process_dataset_sequence(
            dataset_root=dataset_root,
            sequence_idx=0,
            sequence_length=10
        )
        
        print(f"✓ Processed {results['sequence_info']['length']} frames")
        
        # Compute metrics
        pred_poses = results['predicted_poses']
        true_poses = results['true_poses']
        
        # Position error
        pos_error = np.linalg.norm(pred_poses[:, :3] - true_poses[:, :3], axis=1)
        avg_pos_error = np.mean(pos_error)
        
        # Rotation error (quaternion dot product)
        pred_quat = pred_poses[:, 3:] / np.linalg.norm(pred_poses[:, 3:], axis=1, keepdims=True)
        true_quat = true_poses[:, 3:] / np.linalg.norm(true_poses[:, 3:], axis=1, keepdims=True)
        rot_error = np.abs(np.sum(pred_quat * true_quat, axis=1))
        avg_rot_error = np.mean(1 - rot_error)
        
        print(f"Average Position Error: {avg_pos_error:.6f}")
        print(f"Average Rotation Error: {avg_rot_error:.6f}")
        
        # Visualize results
        print("Generating visualization...")
        visualize_results(results, save_path="slam_results.png")
        
        # Test future prediction
        print("\nTesting future prediction...")
        history_length = 6
        left_seq = torch.stack([torch.from_numpy(np.random.randn(2048, 4).astype(np.float32)) 
                               for _ in range(history_length)])
        right_seq = torch.stack([torch.from_numpy(np.random.randn(2048, 4).astype(np.float32)) 
                                for _ in range(history_length)])
        time_deltas = torch.tensor([0.0] + [5e-6] * (history_length - 1))
        
        future_poses = slam_inference.predict_future_trajectory(
            left_seq, right_seq, time_deltas, future_horizon=20e-6
        )
        
        print(f"✓ Predicted {len(future_poses)} future poses")
        print(f"Future trajectory shape: {future_poses.shape}")
        
    except Exception as e:
        print(f"✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()