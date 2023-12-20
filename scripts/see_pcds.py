import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt


from scipy.spatial.distance import cdist

EPIC_WIDTH = 456
EPIC_HEIGHT = 256

FRUSTUM_SIZE = 0.6
FRUSTUM_LINE_RADIUS = 0.02

TRAJECTORY_LINE_RADIUS = 0.02

def get_o3d_FOR(origin=[0, 0, 0],size=10):
    """ 
    Create a FOR that can be added to the open3d point cloud
    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=size)
    mesh_frame.translate(origin)
    return(mesh_frame)

#def draw_pcd_from_path(path1,path2):
#    pcd1 = o3d.io.read_point_cloud(path1)
#    pcd2 = o3d.io.read_point_cloud(path2)
#    o3d.visualization.draw_geometries([pcd1,pcd2])
def draw_pcd_from_path(*args):
    pcds = []
    for pcd in args:
        tmp_pcd = o3d.io.read_point_cloud(pcd)
        pcds.append(tmp_pcd)
    o3d.visualization.draw_geometries(pcds)


############################################################
path_static = "data_tiche/pcds/static_dense_pcd_th_0-1.ply"
path_dyn = "data_tiche/pcds/dyn_dense_pcd_th_0-1.ply"

draw_pcd_from_path(path_static)