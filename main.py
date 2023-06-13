import numpy as np
import math
import taichi as ti
ti.init(arch=ti.vulkan)  # Alternatively, ti.init(arch=ti.cpu)

from components_properties import net_width, net_height
from components_init import *
from rendering import *

# Simulation parameters
dt = 1e-2 / max(net_nodes_width, net_nodes_height) #8e-3
substeps = int(1 / 60 // dt) #1/120
gravity = ti.Vector([0, -9.8, 0])

init_mesh_indices()

@ti.kernel
def init_points():
    init_net()
    init_shackles()
    init_rope_low_bearing()
    init_rope_up_bearing()
    init_ball(ti.Vector([2.5, 15, 1.5]), ti.Vector([0.0, 0.0, 0.0]))

init_points()


# Step


@ti.kernel
def update_vertices():
    net_vertices()
    shackle_vertices()

update_vertices()

# Scene rendering
window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024), vsync=True)
canvas = window.get_canvas()
#canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
current_t = 0.0

while window.running:
    # if current_t > 7.5:
    #     # Reset
    #     init_points()
    #     current_t = 0

    # for i in range(substeps):
    #     #substep()
    #     current_t += dt
    #update_boundary_nodes()
    #update_vertices()

    camera.position(20, 10, 20)
    camera.lookat(7.5, 0.0, 1.5)
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
    scene.lines(rope_vertices, color=(0.5, 0.5, 0.5), width=2)
    
    canvas.scene(scene)
    window.show()