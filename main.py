import numpy as np
import math
import taichi as ti
ti.init(arch=ti.vulkan)  # Alternatively, ti.init(arch=ti.cpu)

from properties import net_width, net_height
from components_barrier import *
from components_fem import *
from rendering import *

# Simulation parameters
# dt = 1.00e-2 / max(net_nodes_width, net_nodes_height) #8e-3 #1e-2 #1.01e-2
# substeps = int(1 / 200 // dt) #1/120 #1/200
dt = 5e-5
substeps = 50 #200

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
    init_ball(ti.Vector([7.5, 1, 1.5]), ti.Vector([0.0, -20.0, 0.0]))

init_points()


@ti.func
def calculate_force_ropes(rid, i, direction):
    force = ti.Vector([0.0, 0.0, 0.0])
    eq_length = 0.08  # rope rest length, < 0.1 is pre-tension
    index = i + 1 if direction == 1 else i - 1
    # 0 is direction = previous
    # 1 is direction = next
    if (direction == 0 and i > 0) or (direction == 1 and i < max_elements - 1 and m_rope[rid, index] != 0.0):
        if 2 <= rid <= 9:
            eq_length = 0.099 # almost no pre-tension for upslope ropes
        if 10 <= rid <= 11:
            eq_length = 0.095 # almost no pre-tension for lateral ropes
        force = fem_spring_damper(x_rope[rid, i], 
                                  x_rope[rid, index], 
                                  v_rope[rid, i], 
                                  v_rope[rid, index], 
                                  rope_spring, 
                                  rope_damper, 
                                  eq_length)
    return force


min_indices = ti.field(dtype=ti.i32, shape=(3, num_shackles_hor))

# Find the rope segment the shackle is attached to
shackle_hor_rope_lb = ti.Vector.field(1, dtype=ti.int64, shape=(n_nets, num_shackles_hor))

# @ti.kernel
# def shackle_rope_segment():
#     for n, sid, i in lb_shackle_distances:
#         current_distance = (x_rope[0, i] - x_shackle_hor[n, sid, 0]).norm()
#         lb_shackle_distances[n, sid, i] = current_distance
#         min_val = current_distance

#         if 0 < i < num_elements_lb_total:  # Ensure valid rope segments
#             # Determine orthogonal projection for both segments attached to the current node
#             dir_i1 = x_rope[0, i+1] - x_rope[0, i]
#             dir_i2 = x_rope[0, i] - x_rope[0, i-1]
            
#             proj_onto_i1 = (dir_i1.normalized().dot(x_shackle_hor[n, sid, 0] - x_rope[0, i])) * dir_i1.normalized()
#             proj_onto_i2 = (dir_i2.normalized().dot(x_shackle_hor[n, sid, 0] - x_rope[0, i-1])) * dir_i2.normalized()

#             proj_point_i1 = x_rope[0, i] + proj_onto_i1
#             proj_point_i2 = x_rope[0, i-1] + proj_onto_i2

#             dist_i1 = (proj_point_i1 - x_shackle_hor[n, sid, 0]).norm()
#             dist_i2 = (proj_point_i2 - x_shackle_hor[n, sid, 0]).norm()

#             # Check if one of these projected distances is a new minimum
#             if dist_i1 < min_val:
#                 min_val = dist_i1
#                 shackle_hor_rope_lb[n, sid] = i  # This segment is defined by nodes i and i+1
                
#             if dist_i2 < min_val:
#                 min_val = dist_i2
#                 shackle_hor_rope_lb[n, sid] = i-1  # This segment is defined by nodes i-1 and i

@ti.kernel
def shackle_rope_segment():
    index_shift = num_elements_lb_horizontal // (num_shackles_hor * 3)
    for n, sid in shackle_hor_rope_lb:
        shackle_hor_rope_lb[n, sid] = num_elements_lb_angled + sid * index_shift + 2 * n * num_shackles_hor

# Step function
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
    for i in ti.grouped(x_rope):
        v_rope[i] += gravity * dt
        v_rope[i] *= ti.exp(-net_drag_damping * dt)
    for i, j in x_post:
        if j == 1:
            v_post[i, j] += gravity * dt

    # # Gravity and air friction
    # v_ball[0] += gravity * dt - air_friction * v_ball[0].norm() * v_ball[0] * dt
    # for i in ti.grouped(x_net):
    #     v_net[i] += gravity * dt - air_friction * v_net[i].norm() * v_net[i] * dt
    # for i in ti.grouped(x_shackle_hor):
    #     v_shackle_hor[i] += gravity * dt - air_friction * v_shackle_hor[i].norm() * v_shackle_hor[i] * dt
    # for i in ti.grouped(x_shackle_ver):
    #     v_shackle_ver[i] += gravity * dt - air_friction * v_shackle_ver[i].norm() * v_shackle_ver[i] * dt
    # for i in ti.grouped(x_rope):
    #     v_rope[i] += gravity * dt - air_friction * v_rope[i].norm() * v_rope[i] * dt
    # for i, j in x_post:
    #     if j == 1:
    #         v_post[i, j] += gravity * dt - air_friction * v_post[i, j].norm() * v_post[i, j] * dt
  
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
                grid_dist_ij = ti.Vector([abs(i[1] - j[1]) * net_quad_size_width,
                                          abs(i[2] - j[2]) * net_quad_size_height])
                original_dist = grid_dist_ij.norm()

                force += fem_spring_damper_yielding(x_net[i], 
                                                    x_net[j], 
                                                    v_net[i], 
                                                    v_net[j], 
                                                    net_spring, 
                                                    net_dashpot_damping, 
                                                    original_dist, 
                                                    net_spring_yield)
        v_net[i] += force * dt / m_net[i]

    # Impulse for horizontal shackles
    for n, i, j in v_shackle_hor:
        force = ti.Vector([0.0, 0.0, 0.0])
        net_node = ti.Vector([n, i * shackle_interval, j * (net_nodes_height - 1)])
        # displacement = x_shackle_hor[n, i, j] - x_net[net_node]
        # correction = displacement / 2 

        # impulse = correction * dt
        # v_net[net_node] += impulse / m_net[net_node]
        # v_shackle_hor[n, i, j] -= impulse / m_shackle_hor[n, i, j]

        # x_net[net_node] += correction
        # x_shackle_hor[n, i, j] -= correction

        force = fem_spring_damper(x_shackle_hor[n, i, j], 
                                    x_net[net_node], 
                                    v_shackle_hor[n, i, j], 
                                    v_net[net_node], 
                                    shackle_spring, 
                                    shackle_damp, 
                                    -1)
        v_net[net_node] -= force * dt / m_net[net_node]
        v_shackle_hor[n, i, j] += force * dt / m_shackle_hor[n, i, j]

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
    for rid, i in x_rope:
        if m_rope[rid, i] == 0:  # Skip uninitialized elements
            continue
        
        # Velocity Verlet
        force_previous = calculate_force_ropes(rid, i, 0) # 0 is direction = previous
        force_next = calculate_force_ropes(rid, i, 1) # 1 is direction = next
        force1 = force_previous + force_next

        a_rope[rid, i] = force1 / m_rope[rid, i]
        # First half-step for velocity
        v_rope[rid, i] += 0.5 * dt * a_rope[rid, i]
        x_rope[rid, i] += dt * v_rope[rid, i]
        # Recalculate forces here based on the new positions
        force_previous = calculate_force_ropes(rid, i, 0)
        force_next = calculate_force_ropes(rid, i, 1)
        force2 = force_previous + force_next
        a_rope[rid, i] = force2 / m_rope[rid, i]
        # Second half-step for velocity
        v_rope[rid, i] += 0.5 * dt * a_rope[rid, i]

        if i == 0 or (m_rope[rid, i + 1] == 0.0):
            if 0 <= rid <= 1: # Upper and lower bearing ropes
                a_rope[rid, i] = ti.Vector([0.0, 0.0, 0.0])
                v_rope[rid, i] = ti.Vector([0.0, 0.0, 0.0])
                if i == 0:
                    x_rope[rid, i] = ti.Vector([-b, 0.0, 0.0])
                if m_rope[rid, i + 1] == 0.0:
                    x_rope[rid, i] = ti.Vector([3*net_width + b, 0.0, 0.0])
            if 2 <= rid < 10: # Upslope ropes
                post_id = (rid - 2) % 4 # Simplified calculation of post_id
                if i == 0:
                    force = fem_spring_damper(x_post[post_id, 1], 
                                               x_rope[rid, i], 
                                               v_post[post_id, 1], 
                                               v_rope[rid, i], 
                                               rope_spring, 
                                               rope_damper, 
                                               -1)
                    v_post[post_id, 1] += force * dt / m_post[post_id, 1]
                    v_rope[rid, i] = v_post[post_id, 1]
                    x_rope[rid, i] = x_post[post_id, 1]

                if m_rope[rid, i + 1] == 0.0:
                    a_rope[rid, i] = ti.Vector([0.0, 0.0, 0.0])
                    v_rope[rid, i] = ti.Vector([0.0, 0.0, 0.0])
                    if 2 <= rid < 6:
                        x_rope[rid, i] = ti.Vector([(rid - 2)*net_width + a/2, h, 0.0])
                    if 6 <= rid < 10:
                        x_rope[rid, i] = ti.Vector([(rid - 6)*net_width - a/2, h, 0.0])
            if rid >= 10: # Lateral support ropes
                post_id = (rid - 10) * 3
                if i == 0:
                    a_rope[rid, i] = ti.Vector([0.0, 0.0, 0.0])
                    v_rope[rid, i] = ti.Vector([0.0, 0.0, 0.0])
                    if rid == 10:
                        x_rope[rid, i] = ti.Vector([-b + shift, 0.0, 0.0])
                    if rid == 11:
                        x_rope[rid, i] = ti.Vector([net_width * 3 + b - shift, 0.0, 0.0])
                if m_rope[rid, i + 1] == 0.0:
                    force = fem_spring_damper(x_post[post_id, 1], 
                                               x_rope[rid, i], 
                                               v_post[post_id, 1], 
                                               v_rope[rid, i], 
                                               rope_spring, 
                                               rope_damper, 
                                               -1)
                    v_post[post_id, 1] += force * dt / m_post[post_id, 1]
                    v_rope[rid, i] = v_post[post_id, 1]
                    x_rope[rid, i] = x_post[post_id, 1]

    # Posts
    for i, j in x_post:
        if j == 1: 
            eq_length = net_height
            force = fem_spring_damper_pin_joint(x_post[i, 0],
                                                x_post[i, j], 
                                                v_post[i, 0], 
                                                v_post[i, j], 
                                                post_spring, 
                                                post_damper, 
                                                eq_length)
            v_post[i, j] += force * dt / m_post[i, j]
            x_post[i, j] += v_post[i, j] * dt
            #x_post[i, j].x = i * net_width


    # Posts - Ropes sliding interaction
    for pid, i in lb_post_distances:
        lb_post_distances[pid, i] = (x_rope[0, i] - x_post[pid, 0]).norm()
        # Check if the rope node is within threshold distance to the post the post
        if lb_post_distances[pid, i] <= 0.1:
            # # Check if the rope node is moving towards the post
            # if (v_rope[0, i].dot(x_post[pid, 0] - x_rope[0, i]) > 0):

            shift = x_post[pid, 0] - x_rope[0, i]
            x_rope[0, i] = x_post[pid, 0]
            velocity_correction = shift / dt
            lerp_factor = 0.5  # Linear interpolation factor
            v_rope[0, i] = (1 - lerp_factor) * v_rope[0, i] + lerp_factor * velocity_correction

    for pid, i in ub_post_distances:
        ub_post_distances[pid, i] = (x_rope[1, i] - x_post[pid, 1]).norm()
        # Check if the rope node is within threshold distance to the post the post
        if ub_post_distances[pid, i] <= 0.1:
            # # Check if the rope node is moving towards the post
            # if (v_rope[1, i].dot(x_post[pid, 1] - x_rope[1, i]) > 0):

            shift = x_post[pid, 1] - x_rope[1, i]
            x_rope[1, i] = x_post[pid, 1]
            velocity_correction = shift / dt
            lerp_factor = 0.5  # Linear interpolation factor
            v_rope[1, i] = (1 - lerp_factor) * v_rope[1, i] + lerp_factor * velocity_correction

                # # Calculate the projection of the rope tension force on the post
                # tension_force = force_previous + force_next  # Assuming these are the forces you calculated earlier
                # post_direction = (x_post[closest_post_id, 0] - x_post[closest_post_id, 1]).normalized()
                # force_on_post = tension_force.dot(post_direction)


    # Ropes - Shackles sliding interaction
    for n, sid in shackle_hor_rope_lb:
        i = shackle_hor_rope_lb[n, sid]
        i_val = int(i[0])

        current_velocity = v_shackle_hor[n, sid, 0]
        remaining_distance = current_velocity.norm() * dt
        final_position = x_shackle_hor[n, sid, 0]

        while remaining_distance > 0:
            # Compute possible movement directions along the rope
            dir_i1 = (x_rope[0, i_val+1] - x_rope[0, i_val]).normalized()
            dir_i2 = (x_rope[0, i_val] - x_rope[0, i_val+1]).normalized()

            proj_dir_i1 = current_velocity.dot(dir_i1)
            proj_dir_i2 = current_velocity.dot(dir_i2)
            proj_dir = 0.0
            segment_length = 0.0
            dir = ti.Vector([0.0, 0.0, 0.0])

            if proj_dir_i1 > proj_dir_i2:
                dir = dir_i1
                proj_dir = proj_dir_i1
                segment_length = (x_rope[0, i_val+1] - x_rope[0, i_val]).norm()
            elif proj_dir_i2 > proj_dir_i1:
                dir = dir_i2
                proj_dir = proj_dir_i2
                segment_length = (x_rope[0, i_val] - x_rope[0, i_val+1]).norm()

            # Friction
            friction_force = -shackle_friction_coefficient * current_velocity
            friction_acceleration = friction_force / m_shackle_hor[n, sid, 0]
            current_velocity += friction_acceleration * dt
            if current_velocity.norm() < 1e-6:  # Avoiding the shackle from moving back due to very low velocities
                break

            # Compute how far the shackle would move in this iteration
            proj_distance = proj_dir * dt
            if proj_distance <= segment_length:
                final_position += dir * proj_distance
                #remaining_distance -= proj_distance
                remaining_distance = 0
                shackle_hor_rope_lb[n, sid] = i_val

                x_shackle_hor[n, sid, 0] = final_position
                v_shackle_hor[n, sid, 0] = current_velocity
                
                # Spring/damper
                force = fem_spring_damper(x_shackle_hor[n, sid, 0], 
                                            x_rope[0, i_val], 
                                            v_shackle_hor[n, sid, 0], 
                                            v_rope[0, i_val], 
                                            shackle_spring, 
                                            shackle_damp, 
                                            -1)
                # v_shackle_hor[n, sid, 0] += force * dt / m_shackle_hor[n, sid, 0]
                # v_rope[0, i_val] -= force * dt / m_rope[0, i_val]

                # x_shackle_hor[n, sid, 0] += v_shackle_hor[n, sid, 0] * dt
                # x_rope[0, i_val] += v_rope[0, i_val] * dt
                #x_shackle_hor[n, sid, 0] = v_shackle_hor[n, sid, 0] * dt

                # # Impulse
                # displacement = x_shackle_hor[n, sid, 0] - x_rope[0, i_val]
                # correction = displacement / 2
                # impulse = correction * dt
                
                # v_rope[0, i_val] += impulse / m_rope[0, i_val]
                # v_shackle_hor[n, sid, 0] -= impulse / m_shackle_hor[n, sid, 0] 

                # x_rope[0, i_val] += correction
                # x_shackle_hor[n, sid, 0] -= correction

                break
            else:
                final_position += dir * segment_length
                remaining_distance -= segment_length

                i_val += 1
                if i_val >= num_elements_lb_total - 1:
                    break


    # for n, sid, i in lb_shackle_distances:
    #     current_distance = (x_rope[0, i] - x_shackle_hor[n, sid, 0]).norm()
    #     lb_shackle_distances[n, sid, i] = current_distance
    #     min_val = 0.1

    #     # Check if the current distance is a new minimum
    #     if current_distance < min_val:
    #         min_val = current_distance
    #         min_indices[n, sid] = i

    # for n, sid in min_indices:
    #     # Initial setup
    #     closest_i = min_indices[n, sid]
    #     impulse_point = ti.Vector([0.0, 0.0, 0.0])
    #     closer_node = 0
    #     farther_node = 0
        
    #     if 0 < closest_i < num_elements_lb_total:
    #         # Determine orthogonal projection
    #         dir_i1 = x_rope[0, closest_i+1] - x_rope[0, closest_i]
    #         dir_i2 = x_rope[0, closest_i] - x_rope[0, closest_i-1]
            
    #         proj_onto_i1 = (dir_i1.normalized().dot(x_shackle_hor[n, sid, 0] - x_rope[0, closest_i])) * dir_i1.normalized()
    #         proj_onto_i2 = (dir_i2.normalized().dot(x_shackle_hor[n, sid, 0] - x_rope[0, closest_i-1])) * dir_i2.normalized()

    #         proj_point_i1 = x_rope[0, closest_i] + proj_onto_i1
    #         proj_point_i2 = x_rope[0, closest_i-1] + proj_onto_i2

    #         dist_i1 = (proj_point_i1 - x_shackle_hor[n, sid, 0]).norm()
    #         dist_i2 = (proj_point_i2 - x_shackle_hor[n, sid, 0]).norm()
            
    #         if dist_i1 < dist_i2:
    #             closer_node = closest_i
    #             farther_node = closest_i + 1
    #             impulse_point = proj_point_i1
    #         else:
    #             closer_node = closest_i - 1
    #             farther_node = closest_i
    #             impulse_point = proj_point_i2

    #         # Calculate displacement and correction
    #         displacement = x_shackle_hor[n, sid, 0] - impulse_point
    #         correction = displacement / 2

    #         # Apply impulse proportional to the distance of the nodes to the projected point
    #         distance_closer = (x_rope[0, closer_node] - impulse_point).norm()
    #         distance_farther = (x_rope[0, farther_node] - impulse_point).norm()

    #         impulse_closer = correction * (distance_farther / (distance_closer + distance_farther))
    #         impulse_farther = correction - impulse_closer

    #         v_rope[0, closer_node] += impulse_closer / m_rope[0, closer_node] * dt
    #         v_rope[0, farther_node] += impulse_farther / m_rope[0, farther_node] * dt
    #         v_shackle_hor[n, sid, 0] -= correction / m_shackle_hor[n, sid, 0] * dt

    #         x_rope[0, closer_node] += impulse_closer
    #         x_rope[0, farther_node] += impulse_farther
    #         x_shackle_hor[n, sid, 0] -= correction

    #         # Calculate the direction of the rope segment and the sliding velocity in that direction
    #         rope_segment_direction = (x_rope[0, farther_node] - x_rope[0, closer_node]).normalized()
    #         sliding_velocity = v_shackle_hor[n, sid, 0].dot(rope_segment_direction) * rope_segment_direction
    #         sliding_velocity = sliding_velocity * shackle_friction_coefficient






    # # Impulse for horizontal shackles
    # for n, i, j in v_shackle_hor:
    #     net_node = ti.Vector([n, i * shackle_interval, j * (net_nodes_height - 1)])
    #     displacement = x_shackle_hor[n, i, j] - x_net[net_node]
    #     correction = displacement / 2 

    #     impulse = correction * dt
    #     v_net[net_node] += impulse / m_net[net_node]
    #     v_shackle_hor[n, i, j] -= impulse / m_shackle_hor[n, i, j]

    #     x_net[net_node] += correction
    #     x_shackle_hor[n, i, j] -= correction

    # for n, sid, i in ub_shackle_distances:
    #     ub_shackle_distances[n, sid, i] = (x_rope[1, i] - x_shackle_hor[n, sid, 1]).norm()
    #     if ub_shackle_distances[n, sid, i][0] <= 0.1:
    #         shift = x_rope[1, i] - x_shackle_hor[n, sid, 1]
    #         x_shackle_hor[n, sid, 1] = x_rope[1, i]
    #         velocity_correction = shift / dt
    #         lerp_factor = 0.7  # Linear interpolation factor
    #         v_shackle_hor[n, sid, 1] = (1 - lerp_factor) * v_shackle_hor[n, sid, 1] + lerp_factor * velocity_correction

    # for n, sid, j in x_shackle_hor:
    #     shackle_pos = x_shackle_hor[n, sid, j]
    #     shackle_vel = v_shackle_hor[n, sid, j]
        
    #     # Identify closest rope points
    #     rope_point1, rope_point2 = get_closest_rope_points(n, shackle_pos)
        
    #     # Ensure rope_point1 is always smaller than rope_point2
    #     if rope_point1 > rope_point2:
    #         rope_point1, rope_point2 = rope_point2, rope_point1
        
    #     rope_direction = (x_rope[n, rope_point2] - x_rope[n, rope_point1]).normalized()
        
    #     # Project velocity
    #     projected_velocity = project_velocity(shackle_vel, rope_direction)
    #     orthogonal_velocity = shackle_vel - projected_velocity
        
    #     # Calculate force between rope and shackle
    #     force_on_shackle = fem_spring_damper(shackle_pos, 
    #                                          x_rope[n, rope_point1], 
    #                                          shackle_vel, 
    #                                          v_rope[n, rope_point1], 
    #                                          rope_spring, 
    #                                          rope_damper)

    #     # Apply forces
    #     if m_shackle_hor[n, sid, j] != 0:  # Avoid division by zero
    #         a_shackle = force_on_shackle / m_shackle_hor[n, sid, j]
    #         v_shackle_hor[n, sid, j] += a_shackle * dt

    #     # If shackle's projected velocity would move it beyond current segment within a timestep, then check the next segment
    #     segment_length = (x_rope[n, rope_point2] - x_rope[n, rope_point1]).norm()
    #     if projected_velocity.norm() * dt > segment_length:
    #         # code to handle the case where the shackle moves beyond the current segment
            
    #         # as a start, you might just stop the shackle's movement or implement more complex logic here
    #         v_shackle_hor[n, sid, j] = ti.Vector([0.0, 0.0, 0.0])

    #     # Subtract orthogonal velocity due to friction
    #     friction_force = -shackle_shackle_friction_coefficient * orthogonal_velocity
    #     v_shackle_hor[n, sid, j] += friction_force / m_shackle_hor[n, sid, j]

    # Final position updates
    x_ball[0] += dt * v_ball[0]
    for i in ti.grouped(x_net):
        #v_net[i] *= ti.exp(-net_drag_damping * dt)
        x_net[i] += v_net[i] * dt 


@ti.kernel
def update_vertices():
    update_net_vertices()
    update_shackle_vertices()
    update_rope_vertices()
    update_post_vertices()

shackle_rope_segment()

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

    camera.position(7.5, 15, 25) #20, 10, 40 #7.5, 15, 35 #30, 5, 5
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