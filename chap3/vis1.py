import open3d

print('visualizing the mesh using open3D')
pcd = open3d.io.read_point_cloud('deform1.ply')
open3d.visualization.draw_geometries([pcd],
                                    mesh_show_wireframe = True)

