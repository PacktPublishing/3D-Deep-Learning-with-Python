import open3d
from pytorch3d.io import load_obj
import torch

mesh_file = "cube_texture.obj"

print('visualizing the mesh using open3D')
mesh = open3d.io.read_triangle_mesh(mesh_file)
open3d.visualization.draw_geometries([mesh],
                                     mesh_show_wireframe = True,
                                     mesh_show_back_face = True,
                                     )

print("Loading the same file with PyTorch3D")
vertices, faces, aux = load_obj(mesh_file)
print('Type of vertices = ', type(vertices))
print("Type of faces = ", type(faces))
print("Type of aux = ", type(aux))
print('vertices = ', vertices)
print('faces = ', faces)
print('aux = ', aux)

texture_images = getattr(aux, 'texture_images')
print('texture_images type = ', type(texture_images))
for key in texture_images:
    print(key)

print(texture_images['Skin'].shape)
