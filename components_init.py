import taichi as ti
from math import pi

from components_properties import *

# Ropes
x_rope = ti.Vector.field(3, dtype=ti.f32, shape=(max_ropes, max_elements))
v_rope = ti.Vector.field(3, dtype=ti.f32, shape=(max_ropes, max_elements))
m_rope = ti.field(dtype=ti.f32, shape=(max_ropes, max_elements))

# @ti.func
# def init_rope(rid: ti.template(), length: ti.template(), start_pos: ti.template(), direction: ti.template()):
#     m_rope.fill(rope_node_mass)
#     num_elements[rid] = length * 10
#     for i in range(num_elements[rid]):
#         x_rope[rid, i] = start_pos + i * 0.1 * direction.normalized()
#         v_rope[rid, i] = ti.Vector([0, 0, 0])

# @ti.func
# def init_rope(rid: ti.template(), length: ti.template(), start_pos: ti.template(), direction: ti.template()):
#     m_rope.fill(rope_node_mass)
    
#     # Calculate the number of elements for each section
#     num_elements[1] = int(length * 3 * 10)  # Adjust multiplier as per your requirement
#     num_elements[2] = int(length * 2 / np.cos(delta) * 10)  # Adjust multiplier as per your requirement
#     num_elements[rid] = num_elements[1] + num_elements[2]

#     # Initialize the horizontal section
#     for i in range(num_elements_horizontal[1]):
#         x_rope[rid, i] = start_pos + i * 0.1 * direction.normalized()
#         v_rope[rid, i] = ti.Vector([0, 0, 0])

#     # Initialize the angled section
#     for i in range(num_elements_horizontal[1], num_elements[rid]):
#         x_rope[rid, i] = x_rope[rid, i - 1] + 0.1 * ti.Vector([np.cos(delta), np.sin(delta), 0])
#         v_rope[rid, i] = ti.Vector([0, 0, 0])


# lower bearing rope
length_lb = net_width * 3 + b * 2
num_elements_lb = round(length_lb * 10)
@ti.func
def init_rope_low_bearing():
    rid = 0
    start_pos = ti.Vector([-b, 0.0, 0.0])
    direction = ti.Vector([1, 0, 0])
    for i in ti.static(range(num_elements_lb)):
        x_rope[rid, i] = start_pos + i * (1 / num_elements_lb) * direction
        v_rope[rid, i] = ti.Vector([0, 0, 0])

# upper bearing rope
length_ub_horizontal = net_width * 3
length_ub_angled = b / ti.cos(delta)
num_elements_ub_horizontal = round(length_ub_horizontal * 10)
num_elements_ub_angled = round(length_ub_angled * 10)
@ti.func
def init_rope_up_bearing():
    rid = 1
    start_pos = ti.Vector([-b, 0.0, 0.0])
    direction_horizontal = ti.Vector([1, 0, 0])
    direction_angled = ti.Vector([ti.cos(delta), ti.sin(delta), 0])
    for i in ti.static(range(num_elements_ub_angled)):
        x_rope[rid, i] = start_pos + i * (1 / num_elements_ub_angled) * direction_angled
        v_rope[rid, i] = ti.Vector([0, 0, 0])
    for i in ti.static(range(num_elements_ub_horizontal)):
        x_rope[rid, i] = ti.Vector([0.0, L, 0.0]) + i * (1 / num_elements_ub_horizontal) * direction_horizontal
        v_rope[rid, i] = ti.Vector([0, 0, 0])
    for i in ti.static(range(num_elements_ub_angled)):
        x_rope[rid, i] = ti.Vector([(net_width * 3), L, 0.0]) + i * (1 / num_elements_ub_angled) * (pi - direction_angled)
        v_rope[rid, i] = ti.Vector([0, 0, 0])

# # upslope ropes
# @ti.func
# def init_rope_up_slope(rid: ti.template(), length: ti.template(), start_pos: ti.template(), direction: ti.template()):
#     m_rope.fill(rope_node_mass)
#     num_elements[rid] = length * 10
#     for i in range(num_elements[rid]):
#         x_rope[rid, i] = start_pos + i * 0.1 * direction.normalized()
#         v_rope[rid, i] = ti.Vector([0, 0, 0])

# # lateral support ropes
# @ti.func
# def init_rope_lat_support(rid: ti.template(), length: ti.template(), start_pos: ti.template(), direction: ti.template()):
#     m_rope.fill(rope_node_mass)
#     num_elements[rid] = length * 10
#     for i in range(num_elements[rid]):
#         x_rope[rid, i] = start_pos + i * 0.1 * direction.normalized()
#         v_rope[rid, i] = ti.Vector([0, 0, 0])

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