import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# # Visualize the model
# def visualize_model(file_path):
#     model = o3d.io.read_point_cloud(file_path)
#     o3d.visualization.draw_geometries([model])

# # Generate spherical camera poses
# def generate_spherical_camera_poses(num_azimuth=16, num_elevation=8, radius=2.0):
#     poses = []
#     for elev_idx in range(num_elevation):
#         elevation = np.pi * (elev_idx / (num_elevation - 1) - 0.5)  # [-90°, 90°] in radians
#         for azimuth_idx in range(num_azimuth):
#             azimuth = 2 * np.pi * azimuth_idx / num_azimuth  # [0°, 360°] in radians

#             # Compute camera position
#             x = radius * np.cos(elevation) * np.cos(azimuth)
#             y = radius * np.sin(elevation)
#             z = radius * np.cos(elevation) * np.sin(azimuth)
#             camera_position = np.array([x, y, z])

#             # Look at the origin
#             look_at = np.array([0, 0, 0])
#             up = np.array([0, 1, 0])  # Camera "up" vector

#             # Compute rotation matrix
#             forward = (look_at - camera_position)
#             forward /= np.linalg.norm(forward)
#             right = np.cross(up, forward)
#             right /= np.linalg.norm(right)
#             up = np.cross(forward, right)

#             # Construct 4x4 transformation matrix
#             pose = np.eye(4)
#             pose[:3, 0] = right
#             pose[:3, 1] = up
#             pose[:3, 2] = forward
#             pose[:3, 3] = camera_position
#             poses.append(pose)
#     return poses

# # Render 2D projections using Matplotlib
# def render_2d_projections_matplotlib(model_path, camera_poses, output_dir="rendered_views"):
#     os.makedirs(output_dir, exist_ok=True)

#     # Load the reconstructed model
#     model = o3d.io.read_point_cloud(model_path)
#     points = np.asarray(model.points)

#     # Render each camera pose
#     for i, pose in enumerate(camera_poses):
#         # Transform points to camera coordinate system
#         transformed_points = (pose[:3, :3] @ points.T).T + pose[:3, 3]

#         # Filter points in front of the camera
#         valid = transformed_points[:, 2] > 0
#         transformed_points = transformed_points[valid]

#         # Project to 2D (perspective projection)
#         fx, fy = 500, 500  # Focal lengths
#         cx, cy = 400, 400  # Principal point
#         projected_points = np.zeros((transformed_points.shape[0], 2))
#         projected_points[:, 0] = fx * transformed_points[:, 0] / transformed_points[:, 2] + cx
#         projected_points[:, 1] = fy * transformed_points[:, 1] / transformed_points[:, 2] + cy

#         # Plot using Matplotlib
#         plt.figure(figsize=(8, 8))
#         plt.scatter(projected_points[:, 0], projected_points[:, 1], s=0.1, c="black")
#         plt.axis("off")
#         plt.savefig(os.path.join(output_dir, f"view_{i:03d}.png"))
#         plt.close()
#         print(f"Saved view {i} to {os.path.join(output_dir, f'view_{i:03d}.png')}")

# # Load NeRF training data
def load_nerf_data(image_dir, pose_path):
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
    images = []
    for file in image_files:
        img = Image.open(file).convert("RGB")
        img = np.array(img) / 255.0  # Normalize to [0, 1]
        images.append(img)
    images = np.stack(images, axis=0)

    poses = np.load(pose_path)

    # Convert to PyTorch tensors
    images = torch.from_numpy(images).float()
    poses = torch.from_numpy(poses).float()

    return images, poses



# reconstructed_model_path = "data/bunny/reconstruction/bun_zipper.ply"

# camera_poses = generate_spherical_camera_poses(num_azimuth=16, num_elevation=8, radius=2.0)
# np.save("camera_poses.npy", camera_poses)

# render_2d_projections_matplotlib(reconstructed_model_path, camera_poses, output_dir="rendered_views")

image_dir = "rendered_views"
pose_path = "camera_poses.npy"
images, poses = load_nerf_data(image_dir, pose_path)
print(f"Loaded {images.shape[0]} images and {poses.shape[0]} poses.")

