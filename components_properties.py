import numpy as np
import math

# General geometries of the net ('Grundlagen zur Qualitätsbeurteilung', S. 22-23)
net_width = 5.0 # 5m
net_height = 3.0 # 3m
n_nets = 3 # number of nets
a = net_width
L = 3
h = 5
f = 0.2
b = 2
epsilon = 0

epsilon = math.radians(epsilon) # convert in radians
alfa = epsilon + math.atan((h - L * np.sin(epsilon)) / (L * np.cos(epsilon) + f))
beta = math.atan((a/2) / np.sqrt((h - L * np.sin(epsilon))**2 + (L * np.cos(epsilon) + f)**2))
delta = math.atan(L / b)

# Ropes
rope_node_mass = 0.5

# Nets
net_nodes_width = 107
net_nodes_height = 64
net_spring = 1e5 #1e5
net_spring_yield = 5e5 # 1.1e5
net_dashpot_damping = 1e4 #1e4
net_drag_damping = 5
net_node_mass = 0.5
net_bending_springs = False # bending springs facilitate bending or deformation by introducing additional connections between nodes

# Shackles
shackle_interval = 5  # create a shackle every 5 net nodes
num_shackles = (net_nodes_width + shackle_interval - 1) // shackle_interval
shackle_spring_pj = 1e6 # pin joint
shackle_damp_pj = 1e3
shackle_spring_sj = 1e4 # sliding joint
shackle_damp_sj = 1e2
shackle_node_mass = 1

# Ball
ball_radius = 0.5
ball_mass = 3000.0 # kg. Reinforced concrete ball with a radius of 10 cm, density of 2400 kg/m^3.

# Collision
collision_spring = 1e6
collision_damping = 150

