import taichi as ti
from math import pi

from properties import *

# Ropes
x_rope = ti.Vector.field(3, dtype=ti.f32, shape=(max_ropes, max_elements))
v_rope = ti.Vector.field(3, dtype=ti.f32, shape=(max_ropes, max_elements))
m_rope = ti.field(dtype=ti.f32, shape=(max_ropes, max_elements))

# lower bearing rope
length_lb_horizontal = net_width * 3
length_lb_angled = ti.sqrt(b**2 + f**2)
num_elements_lb_horizontal = round(length_lb_horizontal * 10)
num_elements_lb_angled = round(length_lb_angled * 10)

@ti.func
def init_rope_bearing_low():
    rid = 0
    start_pos = ti.Vector([-b, 0.0, 0.0])
    direction_angled_up = ti.Vector([b, 0, f])
    direction_horizontal = ti.Vector([net_width * 3, 0, 0])
    direction_angled_down = ti.Vector([b, 0, -f])
    for i in ti.ndrange((num_elements_lb_angled)):
        x_rope[rid, i] = start_pos + (i * 0.1 * direction_angled_up.normalized())
        v_rope[rid, i] = ti.Vector([0, 0, 0])
    offset = num_elements_lb_angled
    for i in ti.ndrange((num_elements_lb_horizontal)):
        x_rope[rid, offset + i] = ti.Vector([0.0, 0.0, f]) + (i * 0.1 * direction_horizontal.normalized())
        v_rope[rid, offset + i] = ti.Vector([0, 0, 0])
    offset += num_elements_lb_horizontal
    for i in ti.ndrange((num_elements_lb_angled)):
        x_rope[rid, offset + i] = ti.Vector([(net_width * 3), 0.0, f]) + (i * 0.1 * direction_angled_down.normalized())
        v_rope[rid, offset + i] = ti.Vector([0, 0, 0])

# upper bearing rope
length_ub_horizontal = net_width * 3
length_ub_angled = ti.sqrt(b**2 + (L + f)**2)
num_elements_ub_horizontal = round(length_ub_horizontal * 10)
num_elements_ub_angled = round(length_ub_angled * 10)

@ti.func
def init_rope_bearing_up():
    rid = 1
    start_pos = ti.Vector([-b, 0.0, 0.0])
    direction_angled_up = ti.Vector([b, 0, (L + f)])
    direction_horizontal = ti.Vector([net_width * 3, 0, 0])
    direction_angled_down = ti.Vector([b, 0, -(L + f)])
    for i in ti.ndrange((num_elements_ub_angled)):
        x_rope[rid, i] = start_pos + (i * 0.1 * direction_angled_up.normalized())
        v_rope[rid, i] = ti.Vector([0, 0, 0])
    offset = num_elements_ub_angled
    for i in ti.ndrange((num_elements_ub_horizontal)):
        x_rope[rid, offset + i] = ti.Vector([0.0, 0.0, (L + f)]) + (i * 0.1 * direction_horizontal.normalized())
        v_rope[rid, offset + i] = ti.Vector([0, 0, 0])
    offset += num_elements_ub_horizontal
    for i in ti.ndrange((num_elements_ub_angled)):
        x_rope[rid, offset + i] = ti.Vector([(net_width * 3), 0.0, (L + f)]) + (i * 0.1 * direction_angled_down.normalized())
        v_rope[rid, offset + i] = ti.Vector([0, 0, 0])

# upslope ropes
length_upslope = ti.sqrt((a/2 - 0)**2 + (h - 0)**2 + (0 - (L + f))**2)
num_elements_upslope = round(length_upslope * 10)
ropes_per_side = 4

@ti.func
def init_rope_upslope():
    start_pos = ti.Vector([0.0, 0.0, L + f])
    direction_right = ti.Vector([a/2, h, -(L + f)])
    direction_left = ti.Vector([-a/2, h, -(L + f)])
    for j in ti.ndrange(ropes_per_side):
        for i in ti.ndrange(num_elements_upslope):
            x_rope[2 + j, i] = start_pos + ti.Vector([j * net_width, 0.0, 0.0]) + (i * 0.1) * direction_right.normalized()
            v_rope[2 + j, i] = ti.Vector([0, 0, 0])
    for v in ti.ndrange(ropes_per_side):
        for w in ti.ndrange(num_elements_upslope):
            x_rope[6 + v, w] = start_pos + ti.Vector([v * net_width, 0.0, 0.0]) + (w * 0.1) * direction_left.normalized()
            v_rope[6 + v, w] = ti.Vector([0, 0, 0])

# lateral support ropes
length_support = ti.sqrt(b**2 + (L + f)**2)
num_elements_support = round(length_support * 10)

@ti.func
def init_rope_support_lat():
    rid = 10
    shift = 0.25
    start_pos_up = ti.Vector([-b + shift, 0.0, 0.0])
    start_pos_down = ti.Vector([(net_width * 3), 0.0, (L + f)])
    direction_up = ti.Vector([b - shift, 0, (L + f)])
    direction_down = ti.Vector([b - shift, 0, -(L + f)])
    for i in ti.ndrange((num_elements_support)):
        x_rope[rid, i] = start_pos_up + (i * 0.1 * direction_up.normalized())
        v_rope[rid, i] = ti.Vector([0, 0, 0])
    offset = num_elements_support
    for i in ti.ndrange((num_elements_support)):
        x_rope[rid, offset + i] = start_pos_down + (i * 0.1 * direction_down.normalized())
        v_rope[rid, offset + i] = ti.Vector([0, 0, 0])


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
        z = j * net_quad_size_height * ti.cos(epsilon) + f
        x_net[n, i, j] = [x, y, z]
        v_net[n, i, j] = [0, 0, 0]

spring_offsets = []
if net_bending_springs:
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                spring_offsets.append(ti.Vector([0, i, j]))
else:
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i, j) != (0, 0):
                spring_offsets.append(ti.Vector([0, i, j]))


# Shackles
x_shackle_hor = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets, num_shackles_hor, 2))
v_shackle_hor = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets, num_shackles_hor, 2))
m_shackle_hor = ti.field(dtype=ti.f32, shape=(n_nets, num_shackles_hor, 2))

x_shackle_ver = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets-1, num_shackles_ver))
v_shackle_ver = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets-1, num_shackles_ver))
m_shackle_ver = ti.field(dtype=ti.f32, shape=(n_nets-1, num_shackles_ver))

@ti.func
def init_shackles():
    m_shackle_hor.fill(shackle_node_mass)
    m_shackle_ver.fill(shackle_node_mass)
    for n, i, j in x_shackle_hor:
        x = n * net_width + i * (net_quad_size_width * shackle_interval)
        y = j * net_height * ti.sin(epsilon)
        z = j * net_height * ti.cos(epsilon) + f
        x_shackle_hor[n, i, j] = [x, y, z]
        v_shackle_hor[n, i, j] = [0, 0, 0]
    for n, i in x_shackle_ver:
        x = (n + 1) * net_width
        y = i * net_height * ti.sin(epsilon)
        z = i * (net_quad_size_height * shackle_interval) * ti.cos(epsilon) + f
        x_shackle_ver[n, i] = [x, y, z]
        v_shackle_ver[n, i] = [0, 0, 0]    


# Ball
x_ball = ti.Vector.field(3, dtype=ti.f32, shape=(1, ))
v_ball = ti.Vector.field(3, dtype=ti.f32, shape=(1, ))
m_ball = ti.field(dtype=ti.f32, shape=())

@ti.func
def init_ball(ball_center: ti.template(), ball_velocity: ti.template()):
    m_ball.fill(ball_mass)
    x_ball[0] = ball_center
    v_ball[0] = ball_velocity