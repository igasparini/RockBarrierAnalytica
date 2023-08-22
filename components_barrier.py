import taichi as ti
from math import pi

from properties import *

##### Ropes
x_rope = ti.Vector.field(3, dtype=ti.f32, shape=(max_ropes, max_elements))
v_rope = ti.Vector.field(3, dtype=ti.f32, shape=(max_ropes, max_elements))
a_rope = ti.Vector.field(3, dtype=ti.f32, shape=(max_ropes, max_elements))
f_rope = ti.Vector.field(3, dtype=ti.f32, shape=(max_ropes, max_elements))
m_rope = ti.field(dtype=ti.f32, shape=(max_ropes, max_elements))

### Lower bearing rope
length_lb_horizontal = net_width * 3
length_lb_angled = ti.sqrt(b**2 + f**2)
num_elements_lb_horizontal = round(length_lb_horizontal * (1/rope_segment_length))
num_elements_lb_angled = round(length_lb_angled * (1/rope_segment_length))
num_elements_lb_total = num_elements_lb_horizontal + num_elements_lb_angled * 2

# distances to posts
lb_post_distances = ti.field(dtype=ti.f32, shape=(4, num_elements_lb_total))

@ti.func
def init_rope_bearing_low():
    rid = 0
    start_pos = ti.Vector([-b, 0.0, 0.0])
    direction_angled_up = ti.Vector([b, 0, f])
    direction_horizontal = ti.Vector([net_width * 3, 0, 0])
    direction_angled_down = ti.Vector([b, 0, -f])
    for i in ti.ndrange((num_elements_lb_angled)):
        x_rope[rid, i] = start_pos + (i * rope_segment_length * direction_angled_up.normalized())
        v_rope[rid, i] = ti.Vector([0, 0, 0])
        a_rope[rid, i] = ti.Vector([0, 0, 0])
        m_rope[rid, i] = rope_node_mass
    offset = num_elements_lb_angled
    for i in ti.ndrange((num_elements_lb_horizontal)):
        x_rope[rid, offset + i] = ti.Vector([0.0, 0.0, f]) + (i * rope_segment_length * direction_horizontal.normalized())
        v_rope[rid, offset + i] = ti.Vector([0, 0, 0])
        a_rope[rid, offset + i] = ti.Vector([0, 0, 0])
        m_rope[rid, offset + i] = rope_node_mass
    offset += num_elements_lb_horizontal
    for i in ti.ndrange((num_elements_lb_angled)):
        x_rope[rid, offset + i] = ti.Vector([(net_width * 3), 0.0, f]) + (i * rope_segment_length * direction_angled_down.normalized())
        v_rope[rid, offset + i] = ti.Vector([0, 0, 0])
        a_rope[rid, offset + i] = ti.Vector([0, 0, 0])
        m_rope[rid, offset + i] = rope_node_mass
    for i, j in lb_post_distances:
        lb_post_distances[i, j] = 0

### Upper bearing rope
length_ub_horizontal = net_width * 3
length_ub_angled = ti.sqrt(b**2 + (L + f)**2)
num_elements_ub_horizontal = round(length_ub_horizontal * (1/rope_segment_length))
num_elements_ub_angled = round(length_ub_angled * (1/rope_segment_length))
num_elements_ub_total = num_elements_ub_horizontal + num_elements_ub_angled * 2

# distances to posts
ub_post_distances = ti.field(dtype=ti.f32, shape=(4, num_elements_ub_total))

@ti.func
def init_rope_bearing_up():
    rid = 1
    start_pos = ti.Vector([-b, 0.0, 0.0])
    direction_angled_up = ti.Vector([b, 0, (L + f)])
    direction_horizontal = ti.Vector([net_width * 3, 0, 0])
    direction_angled_down = ti.Vector([b, 0, -(L + f)])
    for i in ti.ndrange((num_elements_ub_angled)):
        x_rope[rid, i] = start_pos + (i * rope_segment_length * direction_angled_up.normalized())
        v_rope[rid, i] = ti.Vector([0, 0, 0])
        a_rope[rid, i] = ti.Vector([0, 0, 0])
        m_rope[rid, i] = rope_node_mass
    offset = num_elements_ub_angled
    for i in ti.ndrange((num_elements_ub_horizontal)):
        x_rope[rid, offset + i] = ti.Vector([0.0, 0.0, (L + f)]) + (i * rope_segment_length * direction_horizontal.normalized())
        v_rope[rid, offset + i] = ti.Vector([0, 0, 0])
        a_rope[rid, offset + i] = ti.Vector([0, 0, 0])
        m_rope[rid, offset + i] = rope_node_mass
    offset += num_elements_ub_horizontal
    for i in ti.ndrange((num_elements_ub_angled)):
        x_rope[rid, offset + i] = ti.Vector([(net_width * 3), 0.0, (L + f)]) + (i * rope_segment_length * direction_angled_down.normalized())
        v_rope[rid, offset + i] = ti.Vector([0, 0, 0])
        a_rope[rid, offset + i] = ti.Vector([0, 0, 0])
        m_rope[rid, offset + i] = rope_node_mass
    for i, j in ub_post_distances:
        ub_post_distances[i, j] = 0

### Upslope ropes
length_upslope = ti.sqrt((a/2 - 0)**2 + (h - 0)**2 + (0 - (L + f))**2)
num_elements_upslope = round(length_upslope * 1/rope_segment_length) - 1 # -1 to account for rounding errors
ropes_per_side = 4

@ti.func
def init_rope_upslope():
    start_pos = ti.Vector([0.0, 0.0, L + f])
    direction_right = ti.Vector([a/2, h, -(L + f)])
    direction_left = ti.Vector([-a/2, h, -(L + f)])
    for j in ti.ndrange(ropes_per_side):
        for i in ti.ndrange(num_elements_upslope):
            x_rope[2 + j, i] = start_pos + ti.Vector([j * net_width, 0.0, 0.0]) + (i * rope_segment_length) * direction_right.normalized()
            v_rope[2 + j, i] = ti.Vector([0, 0, 0])
            a_rope[2 + j, i] = ti.Vector([0, 0, 0])
            m_rope[2 + j, i] = rope_node_mass
    for v in ti.ndrange(ropes_per_side):
        for w in ti.ndrange(num_elements_upslope):
            x_rope[6 + v, w] = start_pos + ti.Vector([v * net_width, 0.0, 0.0]) + (w * rope_segment_length) * direction_left.normalized()
            v_rope[6 + v, w] = ti.Vector([0, 0, 0])
            a_rope[6 + v, w] = ti.Vector([0, 0, 0])
            m_rope[6 + v, w] = rope_node_mass

### Lateral support ropes
length_support = ti.sqrt((b + rope_lateral_shift)**2 + (L + f)**2)
num_elements_support = round(length_support * (1/rope_segment_length)) - 1 # -1 to account for rounding errors

@ti.func
def init_rope_support_lat():
    rid = 10
    start_pos_left = ti.Vector([-b + rope_lateral_shift, 0.0, 0.0])
    start_pos_right = ti.Vector([(net_width * 3 + b - rope_lateral_shift), 0.0, 0.0])
    direction_left = ti.Vector([b - rope_lateral_shift, 0, (L + f)])
    direction_right = ti.Vector([net_width * 3 - (net_width * 3 + b - rope_lateral_shift), 0, (L + f)])
    for i in ti.ndrange((num_elements_support)):
        x_rope[rid, i] = start_pos_left + (i * rope_segment_length * direction_left.normalized())
        v_rope[rid, i] = ti.Vector([0, 0, 0])
        a_rope[rid, i] = ti.Vector([0, 0, 0])
        m_rope[rid, i] = rope_node_mass
    for j in ti.ndrange((num_elements_support)):
        x_rope[rid + 1, j] = start_pos_right + (j * rope_segment_length * direction_right.normalized())
        v_rope[rid + 1, j] = ti.Vector([0, 0, 0])
        a_rope[rid + 1, j] = ti.Vector([0, 0, 0])
        m_rope[rid + 1, j] = rope_node_mass


##### Nets
net_quad_size_width = net_width / net_nodes_width
net_quad_size_height = net_height / net_nodes_height

x_net = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets, net_nodes_width, net_nodes_height))
v_net = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets, net_nodes_width, net_nodes_height))
f_net = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets, net_nodes_width, net_nodes_height))
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
        f_net[n, i, j] = [0, 0, 0]

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


##### Shackles
x_shackle_hor = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets, num_shackles_hor, 2))
v_shackle_hor = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets, num_shackles_hor, 2))
f_shackle_hor = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets, num_shackles_hor, 2))

x_shackle_ver = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets-1, num_shackles_ver))
v_shackle_ver = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets-1, num_shackles_ver))

@ti.func
def init_shackles():
    for n, i, j in x_shackle_hor:
        x = n * net_width + i * (net_quad_size_width * shackle_interval)
        y = j * net_height * ti.sin(epsilon)
        z = j * net_height * ti.cos(epsilon) + f
        x_shackle_hor[n, i, j] = [x, y, z]
        v_shackle_hor[n, i, j] = [0, 0, 0]
        f_shackle_hor[n, i, j] = [0, 0, 0]

        # Initializing connections
        net_node_idx = ti.Vector([n, i * shackle_interval, j * (net_nodes_height - 1)])
        connections_shackle_hor_net[n, i, j] = net_node_idx
        connections_net_shackle_hor[n, net_node_idx[1], net_node_idx[2]] = [x, y, z]

    for n, i in x_shackle_ver:
        x = (n + 1) * net_width
        y = i * net_height * ti.sin(epsilon)
        z = i * (net_quad_size_height * shackle_interval) * ti.cos(epsilon) + f
        x_shackle_ver[n, i] = [x, y, z]
        v_shackle_ver[n, i] = [0, 0, 0]    


##### Posts
x_post = ti.Vector.field(3, dtype=ti.f32, shape=(num_posts, 2))
v_post = ti.Vector.field(3, dtype=ti.f32, shape=(num_posts, 2))
m_post = ti.field(dtype=ti.f32, shape=(num_posts, 2))

@ti.func
def init_posts():
    m_post.fill(post_node_mass)
    for n, i in x_post:
        x = n * net_width
        y = i * net_height * ti.sin(epsilon)
        z = i * net_height * ti.cos(epsilon) + f
        x_post[n, i] = [x, y, z]
        v_post[n, i] = [0, 0, 0]

##### Ball
x_ball = ti.Vector.field(3, dtype=ti.f32, shape=(1, ))
v_ball = ti.Vector.field(3, dtype=ti.f32, shape=(1, ))
m_ball = ti.field(dtype=ti.f32, shape=())

@ti.func
def init_ball(ball_center: ti.template(), ball_velocity: ti.template()):
    m_ball.fill(ball_mass)
    x_ball[0] = ball_center
    v_ball[0] = ball_velocity





# CONNECTIONS

@ti.func
def find_closest_rope_segment(rid, obj_pos, max_rope_nodes):
    min_distance = 1e6
    best_index = -1
    
    for k in range(max_rope_nodes - 1):
        segment_start = x_rope[rid, k]
        segment_end = x_rope[rid, k + 1]
                
        rope_vector = segment_end - segment_start
        to_obj = obj_pos - segment_start
        projected_length = to_obj.dot(rope_vector) / rope_vector.norm()
                
        if 0 <= projected_length <= rope_vector.norm():
            distance = (segment_start + projected_length * rope_vector.normalized() - obj_pos).norm()
            if distance < min_distance:
                min_distance = distance
                best_index = k
                
    return best_index


##### Connections shackles hor with ropes
connections_shackle_hor_rope = ti.Vector.field(1, dtype=ti.int32, shape=(2, n_nets, num_shackles_hor))
connections_rope_shackle_hor = ti.Vector.field(2, dtype=ti.int32, shape=(2, max(num_elements_lb_horizontal, num_elements_ub_horizontal)))
connections_rope_shackle_hor.fill([99, 99])

@ti.func
def init_connections_shackle_hor_rope():
    for rid, n, sid in connections_shackle_hor_rope:
        shackle_pos = x_shackle_hor[n, sid, rid]
        if rid == 0:  # Lower bearing rope
            max_elements = num_elements_lb_total
        else:  # Upper bearing rope
            max_elements = num_elements_ub_total
        
        i = find_closest_rope_segment(rid, shackle_pos, max_elements)
        connections_shackle_hor_rope[rid, n, sid] = i
        connections_rope_shackle_hor[rid, i] = ti.Vector([n, sid])


##### Connections shackles hor with net
connections_shackle_hor_net = ti.Vector.field(3, dtype=ti.int32, shape=(n_nets, num_shackles_hor, 2))
connections_net_shackle_hor = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets, net_nodes_width, net_nodes_height))
# init of these fields is included in init_shackles


##### Connections posts with bearing ropes
connections_post_rope = ti.Vector.field(1, dtype=ti.int32, shape=(2, num_posts))

@ti.func
def init_connections_post_rope():
    for rid, pid in connections_post_rope:
        post_pos = x_post[pid, rid]
        max_rope_nodes = 0
        if rid == 0:  # Lower bearing rope
            max_rope_nodes = num_elements_lb_total
        else:  # Upper bearing rope
            max_rope_nodes = num_elements_ub_total

        i = find_closest_rope_segment(rid, post_pos, max_rope_nodes)
        connections_post_rope[rid, pid] = i