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
    init_posts()
    init_ball(ti.Vector([7.5, 5, 1.5]), ti.Vector([0.0, 0.0, 0.0]))

init_points()

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

@ti.func
def force(rid, i):

  force = ti.Vector([0.0, 0.0, 0.0])

  # Spring force from previous segment
  if i > 0:
    prev_length = x_rope[rid, i] - x_rope[rid, i-1] 
    prev_norm = prev_length.norm()
    prev_dir = prev_length / prev_norm

    k = rope_spring
    if 2 <= rid <= 9:
       k = rope_spring * (prev_norm - 0.095)  # No pre-tension
    else:
       k = rope_spring * (prev_norm - d)

    force -= k * prev_dir
  
  # Spring force to next segment
  if i < max_elements-1:
   
    next_length = x_rope[rid, i+1] - x_rope[rid, i]
    next_norm = next_length.norm()  
    next_dir = next_length / next_norm

    k = rope_spring 
    if 2 <= rid <= 9:
       k = rope_spring * (next_norm - 0.095) # No pre-tension
    else:
       k = rope_spring * (next_norm - d)

    force -= k * next_dir

  # Damping force 
  velocity = v_rope[rid, i]
  force -= rope_damper * velocity

  return force

# step function
@ti.kernel
def substep():

    # Gravity
    v_ball[0] += gravity * dt
    for i in ti.grouped(x_net):
        v_net[i] += gravity * dt
    for i in ti.grouped(x_shackle_hor):
        v_shackle_hor[i] += gravity * dt
    for i in ti.grouped(x_shackle_ver):
        v_shackle_ver[i] += gravity * dt
    # for i in ti.grouped(x_rope):
    #     v_rope[i] += gravity * dt
    for i, j in x_post:
        if j == 1:
            v_post[i, j] += gravity * dt        

    # Ball collision
    for i in ti.grouped(x_net):
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

    # Net mechanic
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

    # Ropes
    x_rope_prev = x_rope.copy()
    v_rope_prev = v_rope.copy()
    a_rope_prev = a_rope.copy()

    for rid, i in x_rope:
        if i == 0 or (i < max_elements - 1 and m_rope[rid, i + 1] != 0.0):
            continue # Skip pinned ends
        
        # Verlet position update
        x_rope[rid,i] = 2*x_rope[rid,i] - x_prev[rid,i] + (v_prev[rid,i] * dt) + (0.5 * a[rid,i] * dt**2)

        # Compute forces and acceleration
        a[rid,i] = force(rid, i) / m

        # Update velocities
        v_rope[rid,i] = v_prev[rid,i] + 0.5*(a[rid,i] + a_prev[rid,i])*dt 

        # Copy for next step
        x_prev = x_rope.copy() 
        v_prev = v_rope.copy()



    for rid, i in x_rope:
        if m_rope[rid, i] == 0:  # Skip uninitialized elements
            continue
        force = ti.Vector([0.0, 0.0, 0.0])
        #v_rope[rid, i] += gravity * dt
        d = 0.08  # rope rest length, < 0.1 is pre-tension

        # spring force with the previous node in the rope
        if i > 0: 
            length = x_rope[rid, i] - x_rope[rid, i - 1]
            length_norm = length.norm()
            length_direction = length / length_norm if length_norm != 0 else 0
            velocity_difference = v_rope[rid, i] - v_rope[rid, i - 1]
            damping_force = -rope_damper * velocity_difference
            spring_force = -rope_spring * (length_norm - d) * length_direction #abs(length_norm - d) -> with abs, the rope doesn't have compression
            if 2 <= rid <= 9:
                spring_force = -rope_spring * (length_norm - 0.095) * length_direction # no pre-tension for upslope ropes
            force += spring_force + damping_force

        # spring force with the next node in the rope
        if i < max_elements - 1 and m_rope[rid, i + 1] != 0.0:
            length = x_rope[rid, i] - x_rope[rid, i + 1]
            length_norm = length.norm()
            length_direction = length / length_norm if length_norm != 0 else 0
            velocity_difference = v_rope[rid, i] - v_rope[rid, i + 1]
            damping_force = -rope_damper * velocity_difference
            spring_force = -rope_spring * (length_norm - d) * length_direction #abs(length_norm - d)
            if 2 <= rid <= 9:
                spring_force = -rope_spring * (length_norm - 0.095) * length_direction # no pre-tension for upslope ropes
            force += spring_force + damping_force

        if i != 0 or (i < max_elements - 1 and m_rope[rid, i + 1] != 0.0):
            v_rope[rid, i] += gravity * dt
            v_rope[rid, i] += dt * force / m_rope[rid, i]
            x_rope[rid, i] += dt * v_rope[rid, i]

        if rid == 0:
            if i == 0 or m_rope[rid, i + 1] == 0.0:
                v_rope[rid, i] = [0.0, 0.0, 0.0]

        if rid == 1:
            if i == 0 or m_rope[rid, i + 1] == 0.0:
                v_rope[rid, i] = [0.0, 0.0, 0.0]

        

        # if i == 0 or (i < max_elements - 1 and m_rope[rid, i + 1] == 0.0):
        #     if 0 <= rid <= 1:
        #         v_rope[rid, i] = ti.Vector([0.0, 0.0, 0.0])
        #     if 2 <= rid < 10:
        #         post_id = 0
        #         if rid == 2 or rid == 6:
        #             post_id = 0
        #         elif rid == 3 or rid == 7:
        #             post_id = 1
        #         elif rid == 4 or rid == 8:
        #             post_id = 2
        #         elif rid == 5 or rid == 9:
        #             post_id = 3
        #         if i == 0:
        #             force_direction = (x_post[post_id, 1] - x_rope[rid, i]).normalized()
        #             v_post[post_id, 1] -= (force/10000) * force_direction / m_post[post_id, 1]
        #             v_rope[rid, i] = v_post[post_id, 1]
        #         if m_rope[rid, i + 1] == 0.0:
        #             v_rope[rid, i] = ti.Vector([0.0, 0.0, 0.0])
        #     if rid >= 10:
        #         if i == 0:
        #             v_rope[rid, i] = ti.Vector([0.0, 0.0, 0.0])
        #         post_id = (rid - 10) * 3
        #         if m_rope[rid, i + 1] == 0.0:
        #             x_rope[rid, i] = x_post[post_id, 1]
        #     continue

    # Posts
    for i, j in x_post:
        if j == 1:
            d = (x_post[i, 0] - x_post[i, j]).normalized()  # Direction towards the pivot
            displacement = (x_post[i, 0] - x_post[i, j]).norm() - net_height
            spring_force = post_spring * displacement * d
            damping_force = -post_damper * d.dot(v_post[i, j]) * d 
            total_force = spring_force + damping_force
            v_post[i, j] += total_force * dt / m_post[i, j]
            x_post[i, j] += v_post[i, j] * dt
            #x_post[i, j].x = i * net_width


    # Final position updates
    x_ball[0] += dt * v_ball[0]
    for i in ti.grouped(x_net):
        v_net[i] *= ti.exp(-net_drag_damping * dt)
        x_net[i] += v_net[i] * dt 


@ti.kernel
def update_vertices():
    update_net_vertices()
    update_shackle_vertices()
    update_rope_vertices()
    update_post_vertices()

# Scene rendering
window = ti.ui.Window("RockBarrierAnalytica Simulation on GGUI", (1024, 1024), vsync=True)
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

    camera.position(7.5, 15, 35) #20, 10, 40
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
    scene.particles(x_ball, radius=ball_radius * 0.95, color=(0.8, 0, 0))
    scene.particles(shakle_vertices_hor_1, radius=0.05, color=(0, 0.2, 1))
    scene.particles(shakle_vertices_hor_2, radius=0.05, color=(0, 0.2, 1))
    scene.particles(shakle_vertices_hor_3, radius=0.05, color=(0, 0.2, 1))
    scene.particles(shakle_vertices_ver_1, radius=0.05, color=(0, 0.2, 1))
    scene.particles(shakle_vertices_ver_2, radius=0.05, color=(0, 0.2, 1))
    scene.lines(rope_vertices, color=(0.8, 0.8, 0.8), width=2)
    scene.lines(post_vertices, color=(0.3, 0.3, 0.3), width=5)
    
    canvas.scene(scene)
    window.show()