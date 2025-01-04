import open3d as o3d
import matplotlib.pyplot as plt

def visualize_model(file_path):
    model = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([model])

# Example usage
visualize_model("data/bunny/reconstruction/bun_zipper.ply")

# def render_2d_from_reconstructed_model(model_path, save_path, camera_pose):
#     # Load the reconstructed model
#     model = o3d.io.read_point_cloud(model_path)

#     # Setup visualization
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(visible=False)
#     vis.add_geometry(model)

#     # Apply camera pose
#     ctr = vis.get_view_control()
#     ctr.convert_from_pinhole_camera_parameters(camera_pose)

#     # Capture and save the rendered image
#     vis.poll_events()
#     vis.update_renderer()
#     vis.capture_screen_image(save_path)
#     vis.destroy_window()

# # Example: Render a view at 0 degrees
# camera_pose = generate_camera_poses([0])[0]  # Replace with actual pose generation
# render_2d_from_reconstructed_model("data/reconstruction/bun_zipper_res2.ply", "image_000.png", camera_pose)