import taichi as ti

from properties import *
from components_barrier import x_net, x_shackle_hor, x_shackle_ver, x_rope, x_post

# Nets rendering
net_num_triangles = int((net_nodes_width - 1) * (net_nodes_height - 1) * 2)
net_indices = ti.field(int, shape=net_num_triangles * 3)
net_colors = ti.Vector.field(3, dtype=float, shape=net_nodes_width * net_nodes_height)
net_vertices_1 = ti.Vector.field(3, dtype=ti.f32, shape=net_nodes_width * net_nodes_height)
net_vertices_2 = ti.Vector.field(3, dtype=ti.f32, shape=net_nodes_width * net_nodes_height)
net_vertices_3 = ti.Vector.field(3, dtype=ti.f32, shape=net_nodes_width * net_nodes_height)

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

@ti.func
def update_net_vertices():
    for i, j in ti.ndrange(net_nodes_width, net_nodes_height):
        net_vertices_1[i * net_nodes_height + j] = x_net[0, i, j]
        net_vertices_2[i * net_nodes_height + j] = x_net[1, i, j]
        net_vertices_3[i * net_nodes_height + j] = x_net[2, i, j]


# Shackles rendering
shakle_vertices_hor_1 = ti.Vector.field(3, dtype=ti.f32, shape=num_shackles_hor * 2)
shakle_vertices_hor_2 = ti.Vector.field(3, dtype=ti.f32, shape=num_shackles_hor * 2)
shakle_vertices_hor_3 = ti.Vector.field(3, dtype=ti.f32, shape=num_shackles_hor * 2)

shakle_vertices_ver_1 = ti.Vector.field(3, dtype=ti.f32, shape=num_shackles_ver)
shakle_vertices_ver_2 = ti.Vector.field(3, dtype=ti.f32, shape=num_shackles_ver)

@ti.func
def update_shackle_vertices():
    for i, j in ti.ndrange(num_shackles_hor, 2):
        shakle_vertices_hor_1[i * 2 + j] = x_shackle_hor[0, i, j]
        shakle_vertices_hor_2[i * 2 + j] = x_shackle_hor[1, i, j]
        shakle_vertices_hor_3[i * 2 + j] = x_shackle_hor[2, i, j]
    for i, j in ti.ndrange(num_shackles_ver, 2):
        shakle_vertices_ver_1[i] = x_shackle_ver[0, i]
        shakle_vertices_ver_2[i] = x_shackle_ver[1, i]


# Ropes rendering
rope_vertices = ti.Vector.field(3, dtype=ti.f32, shape=max_ropes * max_elements)

@ti.func
def update_rope_vertices():
    for rid, i in ti.ndrange(max_ropes, max_elements):
        rope_vertices[rid * max_elements + i] = x_rope[rid, i]

# Posts rendering
post_vertices = ti.Vector.field(3, dtype=ti.f32, shape= (n_nets + 1) * 2)

@ti.func
def update_post_vertices():
    for pid, i in ti.ndrange(n_nets + 1, 2):
        post_vertices[pid * 2 + i] = x_post[pid, i]