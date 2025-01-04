import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

def parse_conf_file(conf_path):
    transformations = {}

    with open(conf_path, 'r') as file:
        for line in file:
            tokens = line.strip().split()

            if tokens[0] == "bmesh":
                file_name = tokens[1]
                translation = list(map(float, tokens[2:5]))
                quaternion = list(map(float, tokens[5:9]))

                transformations[file_name] = {"translation": translation, "quaternion": quaternion}
    return transformations

conf_path = "data/bunny/data/bun.conf~"
transformations = parse_conf_file(conf_path)
# print(transformations)


def apply_transformation(point_cloud, translation, quaternion):
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    points = np.asarray(point_cloud.points)
    points = np.dot(points, rotation_matrix.T)  # Rotate points

    points += translation

    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud



# Apply transformations to each .ply file
for file_name, transform in transformations.items():

    point_cloud = o3d.io.read_point_cloud(f"data/bunny/data/{file_name}")
    transformed_cloud = apply_transformation(point_cloud, transform["translation"], transform["quaternion"])

    # Save the transformed point cloud
    save_path = f"data/bunny/data/transformed_{file_name}"
    o3d.io.write_point_cloud(save_path, transformed_cloud)



def merge_point_clouds(file_paths):
    merged_cloud = o3d.geometry.PointCloud()
    for file_path in file_paths:
        cloud = o3d.io.read_point_cloud(file_path)
        merged_cloud += cloud
    return merged_cloud


transformed_files = [f"data/bunny/data/transformed_{file_name}" for file_name in transformations.keys()]
merged_cloud = merge_point_clouds(transformed_files)
o3d.io.write_point_cloud("data/bunny/data/merged_bunny.ply", merged_cloud)
# o3d.visualization.draw_geometries([merged_cloud])









