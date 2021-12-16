#!/usr/bin/env python3
#from __future__ import absolute_import, division, print_function, unicode_literals
#import json
#import os
#from os import path
#import cv2
#import numpy as np
#import numpy
#np.set_printoptions(threshold=np.inf)

import open3d
import torch
import pytorch3d
from pytorch3d.io import load_obj
from scipy.spatial.transform import Rotation as Rotation

from pytorch3d.renderer.cameras import PerspectiveCameras, OrthographicCameras

#Load meshes and visualize it with Open3D
mesh_file = "cube.obj"
print('visualizing the mesh using open3D')
mesh = open3d.io.read_triangle_mesh(mesh_file)
'''open3d.visualization.draw_geometries([mesh],
                                     mesh_show_wireframe = True,
                                     mesh_show_back_face = True,
                                     )'''

#Define a mini-batch of 8 cameras
image_size = torch.ones(8, 2)
image_size[:,0] = image_size[:,0] * 1024
image_size[:,1] = image_size[:,1] * 512
image_size = image_size.cuda()

focal_length = torch.ones(8, 2)
focal_length[:,0] = focal_length[:,0] * 1200
focal_length[:,1] = focal_length[:,1] * 300
focal_length = focal_length.cuda()

principal_point = torch.ones(8, 2)
principal_point[:,0] = principal_point[:,0] * 512
principal_point[:,1] = principal_point[:,1] * 256
principal_point = principal_point.cuda()

R = Rotation.from_euler('zyx', [
[n*5, n, n]  for n in range(-4, 4, 1)], degrees=True).as_matrix()
R = torch.from_numpy(R).cuda()
T = [ [n, 0, 0] for n in range(-4, 4, 1)]
T = torch.FloatTensor(T).cuda()

#print('R = ', R)
#exit()

camera = PerspectiveCameras(focal_length = focal_length,
                            principal_point = principal_point,
                            in_ndc = False,
                            image_size = image_size,
                            R = R,
                            T = T,
                            device = 'cuda')

world_to_view_transform = camera.get_world_to_view_transform()
world_to_screen_transform = camera.get_full_projection_transform()
print('The type of world_to_screen_transform = ', type(world_to_screen_transform))

#Load meshes using PyTorch3D
vertices, faces, aux = load_obj(mesh_file)
vertices = vertices.cuda()

world_to_view_vertices = world_to_view_transform.transform_points(vertices)
world_to_screen_vertices = world_to_screen_transform.transform_points(vertices)

print('world_to_view_vertices = ', world_to_view_vertices)
print('world_to_screen_vertices = ', world_to_screen_vertices)

