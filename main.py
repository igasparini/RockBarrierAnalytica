import numpy as np
import taichi as ti
ti.init(arch=ti.vulkan)  # Alternatively, ti.init(arch=ti.cpu)

from components_properties import net_width, net_height
from components_init import *
from rendering import *

# Simulation parameters
dt = 1e-2 / max(net_nodes_width, net_nodes_height) #8e-3
substeps = int(1 / 60 // dt) #1/120
gravity = ti.Vector([0, -9.8, 0])


# Ropes
# Lower bearing rope
init_rope(0, 20, ti.Vector([-7.5, 0.0, 0.0]), ti.Vector([1, 0, 0]))
# Upper bearing rope
init_rope(1, 20, ti.Vector([-7.5, 0.0, net_height]), ti.Vector([1, 0, 0]))


# Nets
net_position = ti.Vector([0.0, 0.0, 0.0])
net_inclination = ti.Vector([1.0, 0.0, 0.0])
# Left net
init_net(0, ti.Vector([0.0, 0.0, 0.0]), ti.Vector([1.0, 0.0, 0.0]))
# Center net
init_net(1, ti.Vector([net_width, 0.0, 0.0]), ti.Vector([1.0, 0.0, 0.0]))
# Right net
init_net(2, ti.Vector([net_width*2, 0.0, 0.0]), ti.Vector([1.0, 0.0, 0.0]))


# Shackles
# Left net
init_shackles(0, ti.Vector([0.0, 0.0, 0.0]), ti.Vector([1.0, 0.0, 0.0]))
# Center net
init_shackles(1, ti.Vector([net_width, 0.0, 0.0]), ti.Vector([1.0, 0.0, 0.0]))
# Right net
init_shackles(2, ti.Vector([net_width*2, 0.0, 0.0]), ti.Vector([1.0, 0.0, 0.0]))


# Ball
x_ball[0] = ti.Vector([2.5, 15, 1.5])
v_ball[0] = ti.Vector([0.0, 0.0, 0.0])



@ti.kernel
def flatten_rope_positions(rid: ti.template(), flattened_positions: ti.ext_arr()):
    for i in range(num_elements[rid]):
        for j in ti.static(range(3)):
            flattened_positions[i * 3 + j] = x_rope[rid, i][j]

flattened_positions = np.zeros((max_elements * 3, ), dtype=np.float32)


# Scene rendering
window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
#canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0

while window.running:
    # if current_t > 7.5:
    #     # Reset
    #     initialize_points()
    #     current_t = 0

    # for i in range(substeps):
    #     substep()
    #     current_t += dt
    # #update_boundary_nodes()
    # update_vertices()

    camera.position(10, 10, 20)
    camera.lookat(2.5, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(net_vertices,
               indices=net_indices,
               per_vertex_color=net_colors,
               two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(x_ball, radius=ball_radius * 0.95, color=(1, 0, 0))
    # For each rope
    for rid in range(max_ropes):
        # Flatten rope positions to a 1D array
        flatten_rope_positions(rid, flattened_positions)
        
        # Convert the 1D numpy array back to a Taichi field
        x_rope_flat = ti.Vector.field(3, dtype=ti.f32, shape=(num_elements[rid]))
        for i in range(num_elements[rid]):
            x_rope_flat[i] = ti.Vector([flattened_positions[i * 3], flattened_positions[i * 3 + 1], flattened_positions[i * 3 + 2]])

        # Render the rope
        scene.lines(x_rope_flat, color=(0.5, 0.5, 0.5), width=2)
    #scene.particles(shackle_pos_field_1D, radius=0.05, color=(0, 0, 1))
    canvas.scene(scene)
    window.show()