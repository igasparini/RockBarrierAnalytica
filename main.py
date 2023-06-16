import numpy as np
import math
import taichi as ti
ti.init(arch=ti.vulkan)  # Alternatively, ti.init(arch=ti.cpu)

from properties import net_width, net_height
from components_init import *
from rendering import *

# Simulation parameters
dt = 1e-2 / max(net_nodes_width, net_nodes_height) #8e-3
substeps = int(1 / 200 // dt) #1/120

init_mesh_indices()

@ti.kernel
def init_points():
    init_net()
    init_shackles()
    init_rope_bearing_low()
    init_rope_bearing_up()
    init_rope_upslope()
    init_rope_support_lat()
    init_ball(ti.Vector([7.5, 5, 1.5]), ti.Vector([0.0, 0.0, 0.0]))

init_points()

# step function
@ti.kernel
def substep():
    for i in ti.grouped(x_net):
        v_net[i] += gravity * dt

    for i in ti.grouped(x_net):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n_nets and 0 <= j[1] < net_nodes_width and 0 <= j[2] < net_nodes_height:
                x_ij = x_net[i] - x_net[j]
                v_ij = v_net[i] - v_net[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()

                grid_dist_ij = ti.Vector([abs(i[1] - j[1]) * net_quad_size_width,
                          abs(i[2] - j[2]) * net_quad_size_height])
                original_dist = grid_dist_ij.norm()

                # Spring force with yielding
                displacement_ratio = current_dist / original_dist - 1
                if displacement_ratio < net_spring_yield / net_spring:
                    force += -net_spring * displacement_ratio * d
                else:
                    force += -net_spring_yield * d
                
                # Dashpot damping
                force += -v_ij.dot(d) * d * net_dashpot_damping * min(net_quad_size_width, net_quad_size_height)

        v_net[i] += force * dt / m_net[i]
        
    for i in ti.grouped(x_net):
        v_net[i] *= ti.exp(-net_drag_damping * dt)
        displacement = x_net[i] - x_ball[0]
        distance = displacement.norm()
        if distance <= ball_radius:
            collision_normal = displacement.normalized()
            penetration_depth = ball_radius - distance
            collision_response_force = collision_spring * penetration_depth ** 2 * collision_normal 
            relative_velocity = v_net[i] - v_ball[0]
            damping_force = collision_damping * relative_velocity.dot(collision_normal) * collision_normal

            x_net[i] += penetration_depth * collision_normal
            v_net[i] += (collision_response_force * dt - damping_force * dt) / m_net[i]
            v_ball[0] -= (collision_response_force * dt - damping_force * dt) / m_ball[None]
    
    v_ball[0].y += gravity.y * dt
    x_ball[0] += dt * v_ball[0]

    # for n, i, j in x_net:
    #     x_net[n, i, j] += v_net[n, i, j] * dt

    # Impulse for horizontal shackles
    for n, i, j in v_shackle_hor:
        net_node = ti.Vector([n, i * shackle_interval, j * (net_nodes_height - 1)])
        displacement = x_shackle_hor[n, i, j] - x_net[net_node]
        correction = displacement / 2 

        impulse = correction * dt
        v_net[net_node] += impulse / m_net[net_node]
        v_shackle_hor[n, i, j] -= impulse / m_shackle_hor[n, i, j]

        x_net[net_node] += correction
        x_shackle_hor[n, i, j] -= correction

    # Impulse for vertical shackles
    for n, i in v_shackle_ver:
        net_node_1 = ti.Vector([n, net_nodes_width - 1, i * shackle_interval])
        net_node_2 = ti.Vector([n + 1, 0, i * shackle_interval])
        middle_point = (x_net[net_node_1] + x_net[net_node_2]) / 2
        displacement_1 = middle_point - x_net[net_node_1]
        displacement_2 = middle_point - x_net[net_node_2]

        impulse_1 = displacement_1 * dt
        impulse_2 = displacement_2 * dt
        v_net[net_node_1] += impulse_1 / m_net[net_node_1]
        v_net[net_node_2] += impulse_2 / m_net[net_node_2]
        v_shackle_ver[n, i] -= (impulse_1 + impulse_2) / m_shackle_ver[n, i]

        x_net[net_node_1] += displacement_1
        x_net[net_node_2] += displacement_2
        x_shackle_ver[n, i] = middle_point

    # for n, i, j in x_net:
    #     x_net[n, i, j] += v_net[n, i, j] * dt

# @ti.func
# def find_segment_for_projection(node_position, current_position, rope_nodes, friction_coefficient, rope_id):
#     # Initialize with current segment
#     start_node = int(current_position)
#     end_node = start_node + 1
#     direction = (rope_nodes[rope_id, end_node] - rope_nodes[rope_id, start_node]).normalized()
#     relative_pos = node_position - rope_nodes[rope_id, start_node]
#     projected_pos = relative_pos.dot(direction) * direction + rope_nodes[rope_id, start_node]

#     # While the projected position is not in the current segment
#     while not (rope_nodes[rope_id, start_node] <= projected_pos <= rope_nodes[rope_id, end_node]):
#         # Calculate the force required to move the shackle to the next segment
#         distance_to_next_segment = ti.sqrt((rope_nodes[rope_id, end_node] - projected_pos).norm_sqr())
#         force_required = distance_to_next_segment * friction_coefficient

#         # If the net node's force is not enough to overcome the friction, break
#         if force_required > 1: #net_node_force: 
#             break

#         # Update to the next segment
#         start_node = end_node
#         end_node += 1
#         direction = (rope_nodes[rope_id, end_node] - rope_nodes[rope_id, start_node]).normalized()
#         projected_pos = relative_pos.dot(direction) * direction + rope_nodes[rope_id, start_node]

#     return start_node, end_node

# @ti.func
# def project_to_rope(n, i, j, rid, rope_x, net_x):
#     current_position = x_shackle_hor[n, i, j]
#     if j == 0:
#         start_node, end_node = find_segment_for_projection(net_x[n, i * shackle_interval, j], current_position, rope_x, shackle_friction_coefficient, rid)
#     else:
#         start_node, end_node = find_segment_for_projection(net_x[n, i * shackle_interval, j * (net_nodes_height - 1)], current_position, rope_x, shackle_friction_coefficient, rid)
#     direction = (rope_x[rid, end_node] - rope_x[rid, start_node]).normalized()
#     if j == 0:
#         relative_pos = net_x[n, i * shackle_interval, j] - rope_x[rid, start_node]
#     else:
#         relative_pos = net_x[n, i * shackle_interval, j * (net_nodes_height - 1)] - rope_x[rid, start_node]
#     projected_pos = relative_pos.dot(direction) * direction + rope_x[rid, start_node]

#     # Update the current position of the shackle on the rope
#     x_shackle_hor[n, i, j] = start_node + projected_pos  # Assuming nodes are equally spaced

#     return projected_pos


@ti.kernel
def update_vertices():
    update_net_vertices()
    update_shackle_vertices()
    update_rope_vertices()

# Scene rendering
window = ti.ui.Window("RockfallBarrierAnalytica Simulation on GGUI", (1024, 1024), vsync=True)
canvas = window.get_canvas()
#canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
current_t = 0.0

while window.running:
    if current_t > 3.5:
        # Reset
        init_points()
        current_t = 0

    for i in range(substeps):
        substep()
        current_t += dt

    update_vertices()

    camera.position(20, 10, 30) #20, 10, 40
    camera.lookat(7.5, 0.0, 1.5) #7.5, 0.0, 1.5
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(net_vertices_1,
               indices=net_indices,
               per_vertex_color=net_colors,
               two_sided=True)
    scene.mesh(net_vertices_2,
               indices=net_indices,
               per_vertex_color=net_colors,
               two_sided=True)
    scene.mesh(net_vertices_3,
               indices=net_indices,
               per_vertex_color=net_colors,
               two_sided=True)
    scene.particles(x_ball, radius=ball_radius * 0.95, color=(1, 0, 0))
    scene.particles(shakle_vertices_hor_1, radius=0.05, color=(0, 0, 1))
    scene.particles(shakle_vertices_hor_2, radius=0.05, color=(0, 0, 1))
    scene.particles(shakle_vertices_hor_3, radius=0.05, color=(0, 0, 1))
    scene.particles(shakle_vertices_ver_1, radius=0.05, color=(0, 0, 1))
    scene.particles(shakle_vertices_ver_2, radius=0.05, color=(0, 0, 1))
    scene.lines(rope_vertices, color=(0, 1, 0), width=2)
    
    canvas.scene(scene)
    window.show()