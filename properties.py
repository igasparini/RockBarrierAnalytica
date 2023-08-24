import taichi as ti
import numpy as np
import math

gravity = ti.Vector([0.0, -9.81, 0.0])
air_friction = 0.002 #0.002

# General geometries of the net ('Grundlagen zur Qualitätsbeurteilung', S. 22-23)
net_width = 5 # 5m
net_height = 3 # 3m
n_nets = 3 # number of nets
n_nets_equivalent = 3
a = net_width
L = 3
h = 5
f = 0.2
b = 3
epsilon = 0

epsilon = math.radians(epsilon) # convert in radians
alfa = epsilon + math.atan((h - L * np.sin(epsilon)) / (L * np.cos(epsilon) + f))
beta = math.atan((a/2) / np.sqrt((h - L * np.sin(epsilon))**2 + (L * np.cos(epsilon) + f)**2))
delta = math.atan(L / b)

# Ropes
rope_segment_length = 0.08 #0.08
rope_node_mass = rope_segment_length #* 10
max_ropes = 14  # maximum number of ropes
max_elements = 300  # assuming a maximum length of 30m for ropes
rope_spring = 1e6 # 4e5 8e5
rope_damper = 1e2 # 3e3 1e4 1.05e4
rope_bending_angle_threshold = 10
rope_bending_constant = 1e3
rope_lateral_shift = 0.25 # shift between bearing ropes and support ropes

# Nets
net_nodes_width = 53 #107
net_nodes_height = 32 #64
net_spring = 1e4 #1e6
net_spring_yield = 5e4 # 5e6
net_dashpot_damping = 1.5e2 #1.5e4
net_drag_damping = 50 #5
net_node_mass = 0.1 #0.05
net_bending_springs = False # bending springs facilitate bending or deformation by introducing additional connections between nodes

# Shackles
shackle_interval = 5  # create a shackle every 5 net nodes
num_shackles_hor = (net_nodes_width + shackle_interval - 1) // shackle_interval
num_shackles_ver = (net_nodes_height + shackle_interval - 1) // shackle_interval
shackle_friction_coefficient = 0.85 #0.85
shackle_spring = 1e6 #1e5
shackle_damper = 5e2
shackle_node_mass = 0.5

# Posts
num_posts = n_nets + 1
post_node_mass = 50 #50
post_spring = 1e8
post_damper = 1e4
post_base_spring = 1e4
post_base_damper = 10

# Ball
ball_radius = 0.5
ball_mass = 500.0 # kg. Reinforced concrete ball with a radius of 10 cm, density of 2400 kg/m^3.

# Collision
collision_spring = 1e6
collision_damper = 150