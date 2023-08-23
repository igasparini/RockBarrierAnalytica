import numpy as np
import math
import taichi as ti
ti.init(arch=ti.vulkan)  # Alternatively, ti.init(arch=ti.cpu)

from properties import net_width, net_height
from components_barrier import *
from components_fem import *
from rendering import *

# Simulation parameters
dt = 5e-5 #5e-5
substeps = 100 #200 #50

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
    init_ball(ti.Vector([7.5, 35, 1.5]), ti.Vector([0.0, 0.0, 0.0]))

    init_connections_shackle_hor_rope()
    init_connections_post_rope()

init_points()


@ti.func
def calculate_force_ropes(rid, i, direction):
    force = ti.Vector([0.0, 0.0, 0.0])
    eq_length = 0.95 * rope_segment_length  # pre-tension bearing ropes
    index = i + 1 if direction == 1 else i - 1
    # 0 is direction = previous
    # 1 is direction = next
    if (direction == 0 and i > 0) or (direction == 1 and i < max_elements - 1 and m_rope[rid, index] != 0.0):
        if 2 <= rid <= 9:
            eq_length = 0.95 * rope_segment_length # pre-tension upslope ropes
        if 10 <= rid <= 11:
            eq_length = 0.95 * rope_segment_length # pre-tension lateral ropes (1 = no pre-tension)
        force = spring_damper_1D(x_rope[rid, i], 
                                  x_rope[rid, index], 
                                  v_rope[rid, i], 
                                  v_rope[rid, index], 
                                  rope_spring, 
                                  rope_damper*2, 
                                  eq_length)
        # force = spring_damper_bending(
        #     x_rope[rid, i-1],
        #     x_rope[rid, i], 
        #     x_rope[rid, index],
        #     x_rope[rid, index+1], 
        #     v_rope[rid, i], 
        #     v_rope[rid, index], 
        #     rope_spring, 
        #     rope_damper,
        #     rope_bending_angle_threshold,
        #     rope_bending_constant,
        #     eq_length)
    return force


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
  
    # Ball collision
    for i in ti.grouped(x_net):
        displacement = x_net[i] - x_ball[0]
        distance = displacement.norm()
        if distance <= ball_radius:
            collision_normal = displacement.normalized()
            penetration_depth = ball_radius - distance
            collision_response_force = collision_spring * penetration_depth ** 2 * collision_normal 
            relative_velocity = v_net[i] - v_ball[0]
            damping_force = collision_damper * relative_velocity.dot(collision_normal) * collision_normal

            x_net[i] += penetration_depth * collision_normal
            v_net[i] += (collision_response_force * dt - damping_force * dt) / m_net[i]
            v_ball[0] -= (collision_response_force * dt - damping_force * dt) / m_ball[None]

    # Net
    for i in ti.grouped(x_net):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n_nets and 0 <= j[1] < net_nodes_width and 0 <= j[2] < net_nodes_height:
                grid_dist_ij = ti.Vector([abs(i[1] - j[1]) * net_quad_size_width,
                                          abs(i[2] - j[2]) * net_quad_size_height])
                original_dist = grid_dist_ij.norm()

                force += spring_damper_yielding(x_net[i], 
                                                    x_net[j], 
                                                    v_net[i], 
                                                    v_net[j], 
                                                    net_spring, 
                                                    net_dashpot_damping, 
                                                    original_dist,
                                                    net_spring_yield)
        
        v_net[i] += force * dt / m_net[i]

    # Vertical shackles
    for n, i in x_shackle_ver:
        net_node_1 = ti.Vector([n, net_nodes_width - 1, i * shackle_interval])
        net_node_2 = ti.Vector([n + 1, 0, i * shackle_interval])
        
        force = spring_damper_no_compression(
            x_net[net_node_1], 
            x_net[net_node_2], 
            v_net[net_node_1], 
            v_net[net_node_2], 
            shackle_spring, 
            shackle_damper, 
            eq_length=0.2)

        middle_point = (x_net[net_node_1] + x_net[net_node_2]) / 2

        v_net[net_node_1] += force / m_net[net_node_1] * dt
        v_net[net_node_2] -= force / m_net[net_node_2] * dt

        x_net[net_node_1] += v_net[net_node_1] * dt
        x_net[net_node_2] += v_net[net_node_2] * dt
        x_shackle_ver[n, i] = middle_point

        # # Old impulse calculation:
        # middle_point = (x_net[net_node_1] + x_net[net_node_2]) / 2
        # displacement_1 = middle_point - x_net[net_node_1]
        # displacement_2 = middle_point - x_net[net_node_2]

        # impulse_1 = displacement_1 * dt
        # impulse_2 = displacement_2 * dt
        # v_net[net_node_1] += impulse_1 / m_net[net_node_1]
        # v_net[net_node_2] += impulse_2 / m_net[net_node_2]
        # v_shackle_ver[n, i] -= (impulse_1 + impulse_2) / shackle_node_mass

        # x_net[net_node_1] += displacement_1
        # x_net[net_node_2] += displacement_2
        # x_shackle_ver[n, i] = middle_point

    # Ropes - horizontal shackles sliding interaction
    slide_along_rope_shackles(
        connections_shackle_hor_rope,
        v_shackle_hor,
        x_shackle_hor,
        v_rope,
        x_rope,
        shackle_node_mass,
        rope_node_mass,
        shackle_friction_coefficient,
        shackle_spring,
        10,
        1, #collision damper
        dt,
        num_elements_lb_total)

    # Ropes
    for rid, i in x_rope:
        if m_rope[rid, i] == 0:  # Skip uninitialized elements
            continue

        force_previous = calculate_force_ropes(rid, i, 0) # 0 is direction = previous
        force_next = calculate_force_ropes(rid, i, 1) # 1 is direction = next
        force = force_previous + force_next #+ f_rope[rid, i]

        v_rope[rid, i] += force / m_rope[rid, i] * dt
        x_rope[rid, i] += v_rope[rid, i] * dt
        f_rope[rid, i] += force

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
                    force = spring_damper(
                        x_post[post_id, 1], 
                        x_rope[rid, i], 
                        v_post[post_id, 1], 
                        v_rope[rid, i], 
                        rope_spring, 
                        rope_damper, 
                        -1)
                    v_post[post_id, 1] += force * dt / post_node_mass
                    x_post[post_id, 1] += v_post[post_id, 1] * dt
                    v_rope[rid, i] -= force * dt / rope_node_mass
                    x_rope[rid, i] += v_rope[rid, i] * dt

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
                        x_rope[rid, i] = ti.Vector([-b + rope_lateral_shift, 0.0, 0.0])
                    if rid == 11:
                        x_rope[rid, i] = ti.Vector([net_width * 3 + b - rope_lateral_shift, 0.0, 0.0])
                if m_rope[rid, i + 1] == 0.0:
                    force = spring_damper(
                        x_post[post_id, 1], 
                        x_rope[rid, i], 
                        v_post[post_id, 1], 
                        v_rope[rid, i], 
                        rope_spring, 
                        rope_damper, 
                        -1)
                    v_post[post_id, 1] += force * dt / post_node_mass
                    x_post[post_id, 1] += v_post[post_id, 1] * dt
                    v_rope[rid, i] -= force * dt / rope_node_mass
                    x_rope[rid, i] += v_rope[rid, i] * dt

    # Ropes - posts sliding interaction
    slide_along_rope_posts(
        connections_post_rope,
        v_post,
        x_post,
        v_rope,
        x_rope,
        post_node_mass,
        rope_node_mass,
        shackle_friction_coefficient,
        1e5,
        100,
        collision_damper,  
        dt,
        num_elements_lb_total)

    # Posts
    for i, j in x_post:
        if j == 1: 
            eq_length = net_height
            force = spring_damper_pin_joint(
                x_post[i, 0],
                x_post[i, j], 
                v_post[i, 0], 
                v_post[i, j], 
                post_spring, 
                post_damper, 
                eq_length)
            # Restrict movement in the x direction (lateral constraints of the pin joint)
            x_eq_position = i * net_width
            x_displacement = x_post[i, j][0] - x_eq_position
            x_velocity = v_post[i, j][0]

            x_spring_force = -post_base_spring * x_displacement
            x_damper_force = -post_base_damper * x_velocity
            x_force = x_spring_force + x_damper_force
            
            v_post[i, j][0] += x_force * dt / m_post[i, j]
            v_post[i, j] += force * dt / m_post[i, j]

            x_post[i, j] += v_post[i, j] * dt

    # Shackle - net interaction
    for n, sid, rid in x_shackle_hor:
        if sid == 0:
            pass
        else:
            net_node = connections_shackle_hor_net[n, sid, rid]
            force = spring_damper(
                x_shackle_hor[n, sid, rid], 
                x_net[net_node[0], net_node[1], net_node[2]], 
                v_shackle_hor[n, sid, rid], 
                v_net[net_node[0], net_node[1], net_node[2]], 
                shackle_spring,
                shackle_damper,  
                -1)
            v_shackle_hor[n, sid, rid] += force * dt / shackle_node_mass
            v_net[net_node[0], net_node[1], net_node[2]] -= force * dt / net_node_mass

            x_shackle_hor[n, sid, rid] += v_shackle_hor[n, sid, rid] * dt
            x_net[net_node[0], net_node[1], net_node[2]] += v_net[net_node[0], net_node[1], net_node[2]] * dt

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