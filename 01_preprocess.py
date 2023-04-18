# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 08:55:35 2023

@author: Julian
"""

# import packages
import laspy as lp
import numpy as np
import matplotlib.pyplot as plt

# read in las file
las = lp.read(r"S:\3D4EcoTec\train\18146.las")

# turn coordinates into numpy array
points = np.stack((las.X, las.Y, las.Z), axis = 1)
points = points * las.header.scale

# set number of pixels
npix = 512

# Recenter data
points = points - np.mean(points, axis = 0)

# Define the camera parameters
fov = 60  # field of view in degrees
near = 0.1  # near clipping plane
far = 100  # far clipping plane
camera_pos = np.array([0, 0, -10])  # camera position
camera_target = np.array([0, 0, 0])  # camera target
up = np.array([0, 1, 0])  # up vector

# Compute the perspective projection matrix
aspect_ratio = 1.0  # assume square pixels
tan_fov = np.tan(np.radians(fov / 2.0))
projection_matrix = np.array([
    [1.0 / (aspect_ratio * tan_fov), 0, 0, 0],
    [0, 1.0 / tan_fov, 0, 0],
    [0, 0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
    [0, 0, -1, 0]])

# Compute the view matrix
#view_direction = camera_target - camera_pos
view_direction = (camera_target - camera_pos) / np.linalg.norm(camera_target - camera_pos)
right = np.cross(view_direction, up)
right /= np.linalg.norm(right)
new_up = np.cross(right, view_direction)
view_matrix = np.eye(4)
view_matrix[:3, :3] = np.vstack((right, new_up, -view_direction))
view_matrix[:3, 3] = -np.dot(view_matrix[:3, :3], camera_pos)

# Compute the projected 2D coordinates
homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
clip_coords = np.dot(np.dot(projection_matrix, view_matrix), homogeneous_points.T).T
clip_coords /= clip_coords[:, 3][:, np.newaxis]
pixel_coords = (clip_coords[:, :2] + 1.0) / 2.0 * np.array([npix, npix])
pixel_coords = pixel_coords[np.argsort(points[:,2])]

"""
# Plot the projected 2D coordinates
plt.scatter(pixel_coords[:, 0], pixel_coords[:, 1], s=z, cmap='viridis')
plt.xlim(0, 800)
plt.ylim(0, 800)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
"""

# Rasterize to a 2D plane with the third dimension as color
fig, ax = plt.subplots()
scatter = ax.scatter(pixel_coords[:, 0], pixel_coords[:, 1], c=np.sort(points[:,2]), cmap='viridis')

# Add a colorbar
#cbar = plt.colorbar(scatter)

plt.xlim(0, npix)
plt.ylim(0, npix)

#cbar.ax.set_ylabel('Z')

# Set the axis labels
#ax.set_xlabel('X')
#ax.set_ylabel('Y')

# Show the plot
plt.show()

"""# Convert the 3D coordinates to pixel coordinates
cols = int((xmax - xmin) / res)
rows = int((ymax - ymin) / res)
x_pix = np.round((points[:, 0] - xmin) / res).astype(int)
y_pix = np.round((ymax - points[:, 1]) / res).astype(int)

# Create an empty raster with the specified bounds and resolution
transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, cols, rows)
raster = np.zeros((rows, cols), dtype=np.uint8)

# Create polygons from the pixel coordinates
polygons = [Polygon([(x, y), (x + res, y), (x + res, y + res), (x, y + res)]) for x, y in zip(x_pix, y_pix)]

# Rasterize the polygons onto the 2D plane
shapes = [(poly, 1) for poly in polygons]
raster = rasterize(shapes, out_shape=raster.shape, transform=transform)

# Plot the raster
plt.imshow(raster, extent=[xmin, xmax, ymin, ymax], cmap='gray', origin='lower')
plt.scatter(x, y, s=1)
plt.show()
"""