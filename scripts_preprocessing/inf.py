import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# === Parameters ===
H, W = 180, 240
step = 2048
camera_size = 0.3  # Size of camera model

# === Load Data ===
left = np.load("/home/adarsh/Documents/SRM/dataset/train/indoor_flying/indoor_flying3/left_Events.npz", allow_pickle=True)
right = np.load("/home/adarsh/Documents/SRM/dataset/train/indoor_flying/indoor_flying3/right_Events.npz", allow_pickle=True)
poses = np.loadtxt("/home/adarsh/Documents/SRM/dataset/train/indoor_flying/indoor_flying3/pose.txt")  # Format: [t tx ty tz qx qy qz qw]

left_events = left['events']
right_events = right['events']

# === Visualization Setup ===
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='6DOF Camera Trajectory Viewer', width=1200, height=800)

# Trajectory path
trajectory_points = []
trajectory_lines = []
line_set = o3d.geometry.LineSet()
vis.add_geometry(line_set)

# Current camera model
current_camera = None

# Store all camera models for trajectory history
camera_history = []

# === Camera Model Creation ===
def create_camera_mesh(size=1.0):
    """Create a 3D camera model using Open3D primitives"""
    # Camera body (main box)
    camera_body = o3d.geometry.TriangleMesh.create_box(
        width=size * 1.5, height=size * 1.0, depth=size * 0.8
    )
    camera_body.translate([-size * 0.75, -size * 0.5, -size * 0.4])
    camera_body.paint_uniform_color([0.3, 0.3, 0.3])  # Dark gray
    
    # Lens (cylinder)
    lens = o3d.geometry.TriangleMesh.create_cylinder(
        radius=size * 0.4, height=size * 0.3
    )
    # Rotate lens to point forward (along +Z axis)
    lens.rotate(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), center=(0, 0, 0))
    lens.translate([0, 0, size * 0.55])
    lens.paint_uniform_color([0.1, 0.1, 0.1])  # Black lens
    
    # Viewfinder (small box on top)
    viewfinder = o3d.geometry.TriangleMesh.create_box(
        width=size * 0.6, height=size * 0.3, depth=size * 0.4
    )
    viewfinder.translate([-size * 0.3, size * 0.5, -size * 0.2])
    viewfinder.paint_uniform_color([0.2, 0.2, 0.2])  # Darker gray
    
    # Camera direction indicator (arrow/pyramid pointing forward)
    direction_indicator = o3d.geometry.TriangleMesh.create_cone(
        radius=size * 0.15, height=size * 0.4
    )
    direction_indicator.rotate(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), center=(0, 0, 0))
    direction_indicator.translate([0, size * 0.7, size * 0.6])
    direction_indicator.paint_uniform_color([1.0, 0.0, 0.0])  # Red arrow
    
    # Combine all parts
    camera_mesh = camera_body + lens + viewfinder + direction_indicator
    
    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size * 0.5)
    camera_mesh += coord_frame
    
    return camera_mesh

def create_simple_camera_wireframe(size=1.0):
    """Create a simple wireframe camera model"""
    # Camera frustum points
    points = np.array([
        # Camera center
        [0, 0, 0],
        # Image plane corners
        [-size, -size * 0.6, size * 1.5],  # Bottom-left
        [size, -size * 0.6, size * 1.5],   # Bottom-right  
        [size, size * 0.6, size * 1.5],    # Top-right
        [-size, size * 0.6, size * 1.5],   # Top-left
    ])
    
    # Lines connecting camera center to image plane corners
    lines = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],  # Center to corners
        [1, 2], [2, 3], [3, 4], [4, 1]   # Image plane rectangle
    ])
    
    # Create line set
    wireframe = o3d.geometry.LineSet()
    wireframe.points = o3d.utility.Vector3dVector(points)
    wireframe.lines = o3d.utility.Vector2iVector(lines)
    wireframe.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]] * len(lines))  # Green
    
    return wireframe

# === Utility Functions ===
def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to rotation matrix using scipy"""
    quat = np.array([qx, qy, qz, qw])  # scipy expects [x, y, z, w] format
    rotation = R.from_quat(quat)
    return rotation.as_matrix()

def create_pose_matrix(tx, ty, tz, qx, qy, qz, qw):
    """Create 4x4 homogeneous transformation matrix from pose"""
    T = np.eye(4)
    T[:3, :3] = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    T[:3, 3] = [tx, ty, tz]
    return T

def events_to_rgb_map(x, y, p):
    """Convert events to RGB polarity map"""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for xi, yi, pi in zip(x, y, p):
        if 0 <= xi < W and 0 <= yi < H:
            if pi > 0:
                img[int(yi), int(xi)] = [0, 0, 255]  # Red for positive events
            elif pi < 0:
                img[int(yi), int(xi)] = [255, 0, 0]  # Blue for negative events
    return img

def add_camera_at_pose(vis, pose_matrix, size=0.2, use_wireframe=True):
    """Add a camera model at the given pose"""
    if use_wireframe:
        camera = create_simple_camera_wireframe(size)
    else:
        camera = create_camera_mesh(size)
    
    camera.transform(pose_matrix)
    vis.add_geometry(camera)
    return camera

# === Main Loop ===
idx = 0
frame_counter = 0

print("Starting 6DOF camera trajectory visualization...")
print("Controls:")
print("- ESC: Exit")
print("- SPACE: Pause/Resume")
print("- 'r': Reset view")
print("- 's': Toggle between solid and wireframe cameras")

paused = False
use_wireframe = True

while idx + step < len(left_events):
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == ord(' '):  # SPACE to pause/resume
        paused = not paused
        continue
    elif key == ord('r'):  # Reset view
        vis.reset_view_point(True)
    elif key == ord('s'):  # Toggle camera style
        use_wireframe = not use_wireframe
        print(f"Switched to {'wireframe' if use_wireframe else 'solid'} camera models")
    
    if paused:
        continue

    # Slice events
    xl, yl, tl, pl = [left_events[k][idx:idx+step] for k in ['x', 'y', 't', 'p']]
    xr, yr, tr, pr = [right_events[k][idx:idx+step] for k in ['x', 'y', 't', 'p']]
    
    if len(xl) == 0 or len(yl) == 0:
        idx += step
        continue

    # Create and display RGB polarity maps
    img_l = events_to_rgb_map(xl, yl, pl)
    img_r = events_to_rgb_map(xr, yr, pr)
    
    # Add frame info to images
    cv2.putText(img_l, f"Left - Frame {frame_counter}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img_r, f"Right - Frame {frame_counter}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add pose info to left image
    pose_idx = min(frame_counter, len(poses) - 1)
    if pose_idx < len(poses):
        _, tx, ty, tz, qx, qy, qz, qw = poses[pose_idx]
        cv2.putText(img_l, f"Pos: ({tx:.2f}, {ty:.2f}, {tz:.2f})", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(img_l, f"Quat: ({qx:.2f}, {qy:.2f}, {qz:.2f}, {qw:.2f})", (10, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    stacked = np.hstack((img_l, img_r))
    cv2.imshow("Event Polarity Maps (Red: ON, Blue: OFF)", stacked)
    
    # Process pose data
    if pose_idx < len(poses):
        # Extract pose: [t tx ty tz qx qy qz qw]
        timestamp, tx, ty, tz, qx, qy, qz, qw = poses[pose_idx]
        
        # Add to trajectory
        current_position = np.array([tx, ty, tz])
        trajectory_points.append(current_position)
        
        # Create pose transformation matrix
        pose_matrix = create_pose_matrix(tx, ty, tz, qx, qy, qz, qw)
        
        # Update current camera model
        if current_camera is not None:
            vis.remove_geometry(current_camera, reset_bounding_box=False)
        
        if use_wireframe:
            current_camera = create_simple_camera_wireframe(camera_size)
        else:
            current_camera = create_camera_mesh(camera_size)
        
        current_camera.transform(pose_matrix)
        vis.add_geometry(current_camera, reset_bounding_box=False)
        
        # Add smaller camera models at intervals for trajectory history
        if frame_counter % 15 == 0 and frame_counter > 0:  # Every 15th frame
            history_camera = add_camera_at_pose(vis, pose_matrix, size=0.15, use_wireframe=True)
            camera_history.append(history_camera)
        
        # Limit history to prevent clutter
        if len(camera_history) > 20:
            old_camera = camera_history.pop(0)
            vis.remove_geometry(old_camera, reset_bounding_box=False)
        
        # Update trajectory lines
        if len(trajectory_points) > 1:
            trajectory_lines.append([len(trajectory_points)-2, len(trajectory_points)-1])
        
        # Update trajectory visualization
        line_set.points = o3d.utility.Vector3dVector(trajectory_points)
        line_set.lines = o3d.utility.Vector2iVector(trajectory_lines)
        
        # Color trajectory: gradient from blue (start) to red (current)
        if len(trajectory_lines) > 0:
            colors = []
            for i, _ in enumerate(trajectory_lines):
                ratio = i / max(1, len(trajectory_lines) - 1)
                # Blue to red gradient
                color = [ratio, 0, 1 - ratio]
                colors.append(color)
            line_set.colors = o3d.utility.Vector3dVector(colors)
        
        vis.update_geometry(line_set)
        
        # Display pose information
        print(f"\rFrame {frame_counter:04d} | Pos: ({tx:.3f}, {ty:.3f}, {tz:.3f}) | "
              f"Cameras: {len(camera_history)+1}", end="")
    
    # Update visualization
    vis.poll_events()
    vis.update_renderer()
    
    idx += step
    frame_counter += 1

print(f"\nVisualization complete. Processed {frame_counter} frames.")
print(f"Total trajectory points: {len(trajectory_points)}")

# Keep window open for final inspection
print("Press any key in the OpenCV window to close...")
cv2.waitKey(0)

# === Cleanup ===
vis.destroy_window()
cv2.destroyAllWindows()