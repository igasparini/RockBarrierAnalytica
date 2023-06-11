import taichi as ti

from components_properties import *

# Ropes
max_ropes = 14  # maximum number of ropes
max_elements = 200  # assuming a maximum length of 20m for ropes

x_rope = ti.Vector.field(3, dtype=ti.f32, shape=(max_ropes, max_elements))
v_rope = ti.Vector.field(3, dtype=ti.f32, shape=(max_ropes, max_elements))
m_rope = ti.field(dtype=ti.f32, shape=(max_ropes, max_elements))

num_elements = ti.field(int, shape=(max_ropes))

@ti.func
def init_rope(rid: ti.template(), length: ti.template(), start_pos: ti.template(), direction: ti.template()):
    m_rope.fill(rope_node_mass)
    num_elements[rid] = length * 10
    for i in range(num_elements[rid]):
        x_rope[rid, i] = start_pos + i * 0.1 * direction.normalized()
        v_rope[rid, i] = ti.Vector([0, 0, 0])

# @ti.func
# def init_rope(rid: ti.template(), length: ti.template(), start_pos: ti.template(), direction: ti.template()):
#     m_rope.fill(rope_node_mass)
    
#     # Calculate the number of elements for each section
#     num_elements_horizontal = int(length * 3 * 10)  # Adjust multiplier as per your requirement
#     num_elements_angled = int(length * 2 / np.cos(delta) * 10)  # Adjust multiplier as per your requirement
#     num_elements[rid] = num_elements_horizontal + num_elements_angled

#     # Initialize the horizontal section
#     for i in range(num_elements_horizontal):
#         x_rope[rid, i] = start_pos + i * 0.1 * direction.normalized()
#         v_rope[rid, i] = ti.Vector([0, 0, 0])

#     # Initialize the angled section
#     for i in range(num_elements_horizontal, num_elements[rid]):
#         x_rope[rid, i] = x_rope[rid, i - 1] + 0.1 * ti.Vector([np.cos(delta), np.sin(delta), 0])
#         v_rope[rid, i] = ti.Vector([0, 0, 0])


# lower bearing rope
@ti.func
def init_rope_low_bearing():

    num_elements[rid] = length * 10
    for i in range(num_elements[rid]):
        x_rope[rid, i] = start_pos + i * 0.1 * direction.normalized()
        v_rope[rid, i] = ti.Vector([0, 0, 0])

# upper bearing rope
@ti.func
def init_rope_up_bearing(rid: ti.template(), length: ti.template(), start_pos: ti.template(), direction: ti.template()):
    m_rope.fill(rope_node_mass)
    num_elements[rid] = length * 10
    for i in range(num_elements[rid]):
        x_rope[rid, i] = start_pos + i * 0.1 * direction.normalized()
        v_rope[rid, i] = ti.Vector([0, 0, 0])

# upslope ropes
@ti.func
def init_rope_up_slope(rid: ti.template(), length: ti.template(), start_pos: ti.template(), direction: ti.template()):
    m_rope.fill(rope_node_mass)
    num_elements[rid] = length * 10
    for i in range(num_elements[rid]):
        x_rope[rid, i] = start_pos + i * 0.1 * direction.normalized()
        v_rope[rid, i] = ti.Vector([0, 0, 0])

# lateral support ropes
@ti.func
def init_rope_lat_support(rid: ti.template(), length: ti.template(), start_pos: ti.template(), direction: ti.template()):
    m_rope.fill(rope_node_mass)
    num_elements[rid] = length * 10
    for i in range(num_elements[rid]):
        x_rope[rid, i] = start_pos + i * 0.1 * direction.normalized()
        v_rope[rid, i] = ti.Vector([0, 0, 0])

# @ti.kernel
# def init_all_ropes():
#     m_rope.fill(rope_node_mass)
#     for rid in range(max_ropes):
#         if rid == 0 or rid == 1:
#             init_special_rope(rid, ... , ... , ...)  # Fill with suitable parameters
#         else:
#             init_rope(rid, ... , ... , ...)  # Fill with suitable parameters

# Nets
net_quad_size_width = net_width / net_nodes_width
net_quad_size_height = net_height / net_nodes_height

x_net = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets, net_nodes_width, net_nodes_height))
v_net = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets, net_nodes_width, net_nodes_height))
m_net = ti.field(dtype=ti.f32, shape=(n_nets, net_nodes_width, net_nodes_height))

@ti.func
def init_net():
    m_net.fill(net_node_mass)
    for n, i, j in x_net:
        x = n * net_width + i * net_quad_size_width
        y = j * net_quad_size_height * ti.sin(epsilon)
        z = j * net_quad_size_height * ti.cos(epsilon)
        x_net[n, i, j] = [x, y, z]
        v_net[n, i, j] = [0, 0, 0]

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
x_shackle = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets, num_shackles, 2))
v_shackle = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets, num_shackles, 2))
m_shackle = ti.field(dtype=ti.f32, shape=(n_nets, num_shackles, 2))

@ti.func
def init_shackles():
    m_shackle.fill(shackle_node_mass)
    for n, i, j in x_shackle:
        x = n * net_width + i * (net_quad_size_width * shackle_interval)
        y = j * net_height * ti.sin(epsilon)
        z = j * net_height * ti.cos(epsilon)
        x_shackle[n, i, j] = [x, y, z]
        v_shackle[n, i, j] = [0, 0, 0]

# Ball
x_ball = ti.Vector.field(3, dtype=ti.f32, shape=(1, ))
v_ball = ti.Vector.field(3, dtype=ti.f32, shape=(1, ))
m_ball = ti.field(dtype=ti.f32, shape=())

@ti.func
def init_ball(ball_center: ti.template(), ball_velocity: ti.template()):
    m_ball.fill(ball_mass)
    x_ball[0] = ball_center
    v_ball[0] = ball_velocity