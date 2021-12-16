import open3d
from pytorch3d.io import load_ply

mesh_file = "cube.ply"
print('visualizing the mesh using open3D')
mesh = open3d.io.read_triangle_mesh(mesh_file)
open3d.visualization.draw_geometries([mesh],
                                     mesh_show_wireframe = True,
                                     mesh_show_back_face = True,
                                     )

print("Loading the same file with PyTorch3D")
vertices, faces = load_ply(mesh_file)
print('Type of vertices = ', type(vertices), ", type of faces = ", type(faces))
print('vertices = ', vertices)
print('faces = ', faces)



