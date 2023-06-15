import numpy as np
import math
import taichi as ti
ti.init(arch=ti.vulkan)  # Alternatively, ti.init(arch=ti.cpu)

from properties import net_width, net_height
from components_init import *
from rendering import *

# Simulation parameters
dt = 1e-2 / max(net_nodes_width, net_nodes_height) #8e-3
substeps = int(1 / 60 // dt) #1/120

init_mesh_indices()

@ti.kernel
def init_points():
    init_net()
    init_shackles()
    init_rope_bearing_low()
    init_rope_bearing_up()
    init_rope_upslope()
    init_rope_support_lat()
    init_ball(ti.Vector([7.5, 15, 1.5]), ti.Vector([0.0, 0.0, 0.0]))

init_points()
init_constants()

@ti.kernel
def substep():
    for i in ti.grouped(x_net):
        v_net[i] += net_gravity[i] * dt

    for i in ti.grouped(x_net):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n_nets and 0 <= j[1] < net_width and 0 <= j[2] < net_height:
                x_ij = x_net[i] - x_net[j]
                v_ij = v_net[i] - v_net[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()

                grid_dist_ij = ti.Vector([abs(i[0] - j[0]) * net_quad_size_width,
                          abs(i[1] - j[1]) * net_quad_size_height])
                original_dist = grid_dist_ij.norm()

                # # Spring force
                # force += -spring_Y * d * (current_dist / original_dist - 1)

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
            collision_response_force = collision_spring * penetration_depth ** 2 * collision_normal  # collision response force
            relative_velocity = v_net[i] - v_ball[0]  # relative velocity of the node and the ball
            damping_force = collision_damping * relative_velocity.dot(collision_normal) * collision_normal  # damping force

            # Update the node's position and velocity
            x_net[i] += penetration_depth * collision_normal
            v_net[i] += (collision_response_force * dt - damping_force * dt) / m_net[i]  # consider mass when updating velocity

            # Update the ball's velocity
            v_ball[0] -= (collision_response_force * dt - damping_force * dt) / m_ball[None]  # consider mass when updating velocity
        
        # # Conditions for the angle points of the net
        # if i[0] != 0 and i[0] != width-1 and i[1] != 0 and i[1] != height-1:
        #     x[i] += dt * v[i]
        # #x[i] += dt * v[i]
    
    # Add motion for the ball
    v_ball[0].y += gravity.y * dt  # Gravity acts on ball
    x_ball[0] += dt * v_ball[0]

    # for i, j in ti.ndrange(width, 2):
    #     force1 = pin_joint(i, j, x[i, j * (height - 1)], v[i, j * (height - 1)])
    #     v[i, j * (height - 1)] += dt * force1 / node_mass[i, j * (height - 1)]

    # # Add sliding_joint forces
    # for i in range(width):
    #     for j in range(2):
    #         force2 = ti.Vector([0.0, 0.0, 0.0])
    #         if j == 0:
    #             force2 = sliding_joint(i, j, upper_bearing_rope_x, upper_bearing_rope_v, x[i, j * (height - 1)], v[i, j * (height - 1)])
    #         else:
    #             force2 = sliding_joint(i, j, lower_bearing_rope_x, lower_bearing_rope_v, x[i, j * (height - 1)], v[i, j * (height - 1)])
            
    #         v[i, j * (height - 1)] += dt * force2 / node_mass[i, j * (height - 1)]
    #         shackle_velocity[i, j] += dt * force2 / node_mass[i, j * (height - 1)]

    # # Update position
    # for i, j in x:
    #     x[i, j] += dt * v[i, j]
    #     if j == 0 or j == height-1:
    #         shackle_position[i, j // (height - 1)] = upper_bearing_rope_x[i] if j == 0 else lower_bearing_rope_x[i]

    # for i in range(width):
    #     for j in range(2):
    #         shackle_pos_field_1D[i * 2 + j] = shackle_position[i, j]

    for i in ti.grouped(x_net):
        x_net[i] += dt * v_net[i]

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
    if current_t > 7.5:
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
    scene.particles(shakle_vertices_1, radius=0.05, color=(0, 0, 1))
    scene.particles(shakle_vertices_2, radius=0.05, color=(0, 0, 1))
    scene.particles(shakle_vertices_3, radius=0.05, color=(0, 0, 1))
    scene.lines(rope_vertices, color=(0, 1, 0), width=2)
    
    canvas.scene(scene)
    window.show()