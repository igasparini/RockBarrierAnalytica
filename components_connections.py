import taichi as ti
from math import pi

from properties import *
# from components_barrier import *

# ##### Connections shackles hor with ropes
# connections_shackle_hor_rope = ti.Vector.field(1, dtype=ti.int32, shape=(2, n_nets, num_shackles_hor))
# connections_rope_shackle_hor = ti.Vector.field(2, dtype=ti.int32, shape=(2, max(num_elements_lb_horizontal, num_elements_ub_horizontal)))
# connections_rope_shackle_hor.fill([99, 99])

# @ti.func
# def find_closest_segment(rid, shackle_pos, max_elements):
#     min_distance = 1e6
#     best_index = -1
    
#     for k in range(max_elements - 1):
#         segment_start = x_rope[rid, k]
#         segment_end = x_rope[rid, k + 1]
                
#         rope_vector = segment_end - segment_start
#         to_shackle = shackle_pos - segment_start
#         projected_length = to_shackle.dot(rope_vector) / rope_vector.norm()
                
#         if 0 <= projected_length <= rope_vector.norm():
#             distance = (segment_start + projected_length * rope_vector.normalized() - shackle_pos).norm()
#             if distance < min_distance:
#                 min_distance = distance
#                 best_index = k
                
#     return best_index

# @ti.func
# def init_connections_shackle_hor_rope():
#     for rid, n, sid in connections_shackle_hor_rope:
#         shackle_pos = x_shackle_hor[n, sid, rid]
#         if rid == 0:  # Lower bearing rope
#             max_elements = num_elements_lb_total
#         else:  # Upper bearing rope
#             max_elements = num_elements_ub_total
        
#         i = find_closest_segment(rid, shackle_pos, max_elements)
#         connections_shackle_hor_rope[rid, n, sid] = i
#         connections_rope_shackle_hor[rid, i] = ti.Vector([n, sid])

# ##### Connections shackles hor with net
# connections_shackle_hor_net = ti.Vector.field(3, dtype=ti.int32, shape=(n_nets, num_shackles_hor, 2))
# connections_net_shackle_hor = ti.Vector.field(3, dtype=ti.f32, shape=(n_nets, net_nodes_width, net_nodes_height))
# # init of these fields is included in init_shackles
