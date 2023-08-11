import taichi as ti

from properties import *
from components_barrier import *

@ti.func
def fem_spring_damper(x1, x2, v1, v2, spring_const, damper_const, eq_length=-1):
    """
    Models a simple spring/damper system and returns the force.

    eq_ength = equilibrium length of the spring

    """
    displacement = x1 - x2
    displacement_norm = displacement.norm()
    displacement_direction = displacement / displacement_norm if displacement_norm != 0 else 0
    velocity_difference = v1 - v2

    if eq_length == -1:
        eq_length = 0

    spring_force = -spring_const * (displacement_norm - eq_length) * displacement_direction
    #damping_force = -damper_const * velocity_difference # 1D damping
    damping_force = -damper_const * velocity_difference.dot(displacement_direction) * displacement_direction #* min(net_quad_size_width, net_quad_size_height) # dashpot damping

    force = spring_force + damping_force
    return force

@ti.func
def fem_spring_damper_1D(x1, x2, v1, v2, spring_const, damper_const, eq_length=-1):
    """
    Models a simple spring/damper system and returns the force.

    eq_ength = equilibrium length of the spring

    """
    displacement = x1 - x2
    displacement_norm = displacement.norm()
    displacement_direction = displacement / displacement_norm if displacement_norm != 0 else 0
    velocity_difference = v1 - v2

    if eq_length == -1:
        eq_length = 0

    spring_force = -spring_const * (displacement_norm - eq_length) * displacement_direction
    #damping_force = -damper_const * velocity_difference # 1D damping
    damping_force = -damper_const * velocity_difference.dot(displacement_direction) * displacement_direction #* min(net_quad_size_width, net_quad_size_height) # dashpot damping

    force = spring_force + damping_force
    return force

@ti.func
def fem_spring_damper_yielding(x1, x2, v1, v2, spring_const, damper_const, eq_length=-1, yield_force=0):
    """
    Models a spring/damper system that allows yelding of the spring and returns the force.

    eq_ength = equilibrium length of the spring
    yield_force = force from which the spring esibites plastic behaviour

    """
    displacement = x1 - x2
    displacement_norm = displacement.norm()
    displacement_direction = displacement.normalized()
    velocity_difference = v1 - v2

    if eq_length == -1.0:
        eq_length = 0

    spring_force = ti.Vector([0.0, 0.0, 0.0])

    displacement_ratio = displacement_norm / eq_length - 1
    if displacement_ratio < yield_force / spring_const:
        spring_force = -spring_const * displacement_ratio * displacement_direction
    else:
        spring_force = -yield_force * displacement_direction
    
    damping_force = -velocity_difference.dot(displacement_direction) * displacement_direction * damper_const * min(net_quad_size_width, net_quad_size_height)

    force = spring_force + damping_force
    return force

@ti.func
def fem_spring_damper_pin_joint(x1, x2, v1, v2, spring_const, damper_const, eq_length=-1):
    """
    Models a simple spring/damper system and returns the force.

    eq_ength = equilibrium length of the spring

    """
    displacement = x1 - x2
    displacement_norm = displacement.norm()
    displacement_direction = displacement.normalized() #displacement / displacement_norm if displacement_norm != 0 else 0 # equal to displacement.normalized()
    velocity_difference = v1 - v2

    if eq_length == -1:
        eq_length = 0

    spring_force = spring_const * (displacement_norm - eq_length) * displacement_direction
    damping_force = -damper_const * displacement_direction.dot(v2) * displacement_direction #* min(net_quad_size_width, net_quad_size_height) # dashpot damping

    force = spring_force + damping_force
    return force