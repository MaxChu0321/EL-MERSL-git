import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from skimage import measure
import pickle
import cv2
import open3d as o3d
from scipy.ndimage import zoom
import trimesh



def find_contours(label_img:np.array):
    mask = ~np.all(label_img == label_img[..., :1], axis=-1) # 如果 RGB 不相等回傳 True
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    else: return contours


def getMask(contour, shape):
    label_img = np.zeros(shape, dtype='uint8')
    label_img = cv2.drawContours(label_img,contour, -1, 1, thickness=cv2.FILLED)
    return label_img


def calc_vol(masks:np.array, slice_loc:list, spacing_xy=tuple, rescale=1) -> np.array:
    '''  Create Voxel Grid  
    Args:
        mask: image mask (2d mask * slices), 3-d, 0/1
        slice_location: each slice z height (unit=mm)
        spacing_xy: unit=mm
        rescale: target voxel spacing, unit=mm
    Returns:
        vol: Voxel Grid, each voxel spacing is rescale
    '''
    slice_thickness = spacing_xy
    # Create 3D grid for interpolation
    z = np.array(slice_loc)
    y = np.arange(0, masks[0].shape[0] * spacing_xy, spacing_xy)
    x = np.arange(0, masks[0].shape[1] * spacing_xy, spacing_xy)

    # Convert sorted masks to binary and stack along the z-axis
    masks_bin = np.array([(mask > 0).astype(int) for mask in masks])

    # Create 3D grid for the masks
    z_mask = np.arange(z.min(), z.max() + slice_thickness, slice_thickness)
    x_mask, y_mask, z_mask = np.meshgrid(x, y, z_mask, indexing='ij')

    # Create the interpolator function
    interpolator = interpolate.RegularGridInterpolator((x, y, z), np.moveaxis(masks_bin, 0, -1), method='nearest', bounds_error=False, fill_value=0)    

    # Interpolate to fill missing slices
    vol = interpolator((x_mask, y_mask, z_mask))

    if rescale != None:
        original_spacing = np.repeat(spacing_xy, 3)
        target_spacing = np.repeat(rescale, 3)
        zoom_factor = original_spacing / target_spacing
        vol = zoom(vol, zoom_factor, order=0)
        vol = vol >=0.5

    return vol


def calc_mesh(vol: np.array)->o3d.geometry.TriangleMesh:
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    
    return mesh


def smooth_mesh(mesh:o3d.geometry.TriangleMesh, pcd_samples=3000, poisson_depth=8)->o3d.geometry.TriangleMesh:
    pcd = mesh.sample_points_poisson_disk(pcd_samples)
    smoothed_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=poisson_depth)
    smoothed_mesh.compute_vertex_normals()
    
    return smoothed_mesh


def plot_mesh(mesh:o3d.geometry.TriangleMesh, color="red", alpha=0.8):
    o3d_vert = np.asarray(mesh.vertices)
    o3d_face = np.asarray(mesh.triangles)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    ax.plot_trisurf(o3d_vert[:, 0], o3d_vert[:, 1], o3d_face, o3d_vert[:, 2],
                    linewidth=0.2, antialiased=True, color=color, alpha=alpha)

    plt.show()


def plot_mesh3d(verts:np.array, faces:np.array):
    tri_mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    tri_mesh.show(file_type='png', filename="./tmp.png")