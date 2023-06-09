import taichi as ti
ti.init(arch=ti.vulkan) 

from components_properties import *
from components_init import *

# Nets rendering
net_num_triangles = int((net_nodes_width - 1) * (net_nodes_height - 1) * 2)
net_indices = ti.field(int, shape=net_num_triangles * 3)
net_vertices = ti.Vector.field(3, dtype=ti.f32, shape=net_nodes_width * net_nodes_height)
net_vertices.fill(1)
net_colors = ti.Vector.field(3, dtype=float, shape=net_nodes_width * net_nodes_height)

@ti.kernel
def init_mesh_indices(): # since Taichi only supports the rendering of triangles, we have to build squares by putting 2 triangles together
    for i, j in ti.ndrange(net_nodes_width - 1, net_nodes_height - 1):
        quad_id = (i * (net_nodes_height - 1)) + j
        # 1st triangle of the square
        net_indices[quad_id * 6 + 0] = i * net_nodes_height + j
        net_indices[quad_id * 6 + 1] = (i + 1) * net_nodes_height + j
        net_indices[quad_id * 6 + 2] = i * net_nodes_height + (j + 1)
        # 2nd triangle of the square
        net_indices[quad_id * 6 + 3] = (i + 1) * net_nodes_height + j + 1
        net_indices[quad_id * 6 + 4] = i * net_nodes_height + (j + 1)
        net_indices[quad_id * 6 + 5] = (i + 1) * net_nodes_height + j

    for i, j in ti.ndrange(net_nodes_width, net_nodes_height):
        if (i // 4 + j // 4) % 2 == 0:
            net_colors[i * net_nodes_height + j] = (0.8, 0.8, 0.8)
        else:
            net_colors[i * net_nodes_height + j] = (0.5, 0.5, 0.5)

# @ti.kernel
# def update_vertices(nid: ti.i32):
#     for i, j in ti.ndrange(net_nodes_width, net_nodes_height):
#         net_vertices[nid, i * net_nodes_height + j] = x_net[nid, i, j]
