import open3d as o3d

def visualize_ply(file_path):
    ply_data = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([ply_data])

# Example usage
visualize_ply("data/bunny/data/bun090.ply")