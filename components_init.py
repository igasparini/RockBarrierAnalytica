import taichi as ti

from components_properties import *

# Ropes
max_ropes = 14  # maximum number of ropes
max_elements = 200  # assuming a maximum length of 20m for ropes

x_rope = ti.Vector.field(3, dtype=ti.f32, shape=(max_ropes, max_elements))
v_rope = ti.Vector.field(3, dtype=ti.f32, shape=(max_ropes, max_elements))
m_rope = ti.field(dtype=ti.f32, shape=(max_ropes, max_elements))
m_rope.fill(rope_node_mass)

num_elements = ti.field(int, shape=(max_ropes))

@ti.func
def init_rope(rid: ti.template(), length: ti.template(), start_pos: ti.template(), direction: ti.template()):
    num_elements[rid] = length * 10
    for i in range(num_elements[rid]):
        x_rope[rid, i] = start_pos + i * 0.1 * direction.normalized()
        v_rope[rid, i] = ti.Vector([0, 0, 0])

# Nets
max_nets = 3  # maximum number of nets

net_quad_size_width = net_width / net_nodes_width
net_quad_size_height = net_height / net_nodes_height

x_net = ti.Vector.field(3, dtype=ti.f32, shape=(max_nets, net_nodes_width, net_nodes_height))
v_net = ti.Vector.field(3, dtype=ti.f32, shape=(max_nets, net_nodes_width, net_nodes_height))
m_net = ti.field(dtype=ti.f32, shape=(max_nets, net_nodes_width, net_nodes_height))
m_net.fill(net_node_mass)

@ti.func
def init_net(nid: ti.template(), start_pos: ti.template(), direction: ti.template()):
    direction_norm = direction.normalized()
    # Assume direction has non-zero x and z components for simplicity
    # y-axis direction is cross product of direction and [0, 1, 0]
    y_dir = ti.Vector([0, 1, 0]).cross(direction_norm).normalized()
    x_dir = y_dir.cross(direction_norm).normalized()  # x-axis direction
    for I in ti.grouped(x_net):
        pos = start_pos + I[0] * net_quad_size_width * x_dir + I[1] * net_quad_size_height * direction_norm
        x_net[nid, I[0], I[1]] = pos
        v_net[nid, I[0], I[1]] = [0, 0, 0]

spring_offsets = []
if net_bending_springs:
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                spring_offsets.append(ti.Vector([i, j]))
else:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([i, j]))

# Shackles
shackle_interval = 5  # create a shackle every 5 net nodes
num_shackles = (net_nodes_width + shackle_interval - 1) // shackle_interval

x_shackle = ti.Vector.field(3, dtype=ti.f32, shape=(max_nets, num_shackles, 2))
v_shackle = ti.Vector.field(3, dtype=ti.f32, shape=(max_nets, num_shackles, 2))

height_values = ti.field(dtype=ti.i32, shape=2)
height_values[0] = 0
height_values[1] = net_nodes_height - 1

@ti.func
def init_shackles(sid: ti.template(), start_pos: ti.template(), direction: ti.template()):
    direction_norm = direction.normalized()
    # Assume direction has non-zero x and z components for simplicity
    # y-axis direction is cross product of direction and [0, 1, 0]
    y_dir = ti.Vector([0, 1, 0]).cross(direction_norm).normalized()
    x_dir = y_dir.cross(direction_norm).normalized()  # x-axis direction
    
    for i in range(0, num_shackles):
        # Compute the shackle position at the bottom and top of the net
        for j in range(2):  # use range(2) instead of the list
            pos = start_pos + i * shackle_interval * net_quad_size_width * x_dir + height_values[j] * net_quad_size_height * direction_norm
            x_shackle[sid, i, j] = pos
            v_shackle[sid, i, j] = [0, 0, 0]

# Ball
x_ball = ti.Vector.field(3, dtype=ti.f32, shape=(1, ))
v_ball = ti.Vector.field(3, dtype=ti.f32, shape=(1, ))
m_ball = ti.field(dtype=ti.f32, shape=())
m_ball.fill(ball_mass)
