import taichi as ti

from properties import *
from components_barrier import *

@ti.func
def spring_damper(x1, x2, v1, v2, spring_const, damper_const, eq_length=-1):
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
def spring_damper_1D(x1, x2, v1, v2, spring_const, damper_const, eq_length=-1):
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
def spring_damper_yielding(x1, x2, v1, v2, spring_const, damper_const, eq_length=-1, yield_force=0):
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
def spring_damper_pin_joint(x1, x2, v1, v2, spring_const, damper_const, eq_length=-1):
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

@ti.func
def slide_along_rope_posts(connections_obj_rope, v_obj, x_obj, v_rope, x_rope, obj_mass, rope_mass, obj_friction_coeff, obj_spring, obj_damper, collision_damper_coeff, dt, num_elements_total):
    """
    Models the interaction of an object sliding along a rope.

    """
    for rid, obj_id in connections_obj_rope:
        i = connections_obj_rope[rid, obj_id]
        i_val = int(i[0])

        current_velocity = v_obj[obj_id, rid]
        remaining_distance = current_velocity.norm() * dt
        final_position = x_obj[obj_id, rid]

        while remaining_distance > 0:
            # Compute possible movement directions along the rope
            dir_i1 = (x_rope[rid, i_val+1] - x_rope[rid, i_val]).normalized()
            dir_i2 = (x_rope[rid, i_val] - x_rope[rid, i_val+1]).normalized()

            proj_dir_i1 = current_velocity.dot(dir_i1)
            proj_dir_i2 = current_velocity.dot(dir_i2)
            proj_dir = 0.0
            segment_length = (x_rope[rid, i_val+1] - x_rope[rid, i_val]).norm()
            dir = ti.Vector([0.0, 0.0, 0.0])

            # Logic to determine direction
            if proj_dir_i1 > proj_dir_i2:
                dir = dir_i1
                proj_dir = proj_dir_i1
            elif proj_dir_i2 > proj_dir_i1:
                dir = dir_i2
                proj_dir = proj_dir_i2

            # Friction
            friction_force = -obj_friction_coeff * current_velocity
            friction_acceleration = friction_force / obj_mass
            current_velocity += friction_acceleration * dt

            # Avoiding the object from moving back due to very low velocities
            if current_velocity.norm() < 1e-6:
                break

            # Compute how far the object would move in this iteration
            proj_distance = proj_dir * dt

            # Object collision handling logic
            displacement = x_obj[obj_id+1, rid] - x_obj[obj_id, rid]
            distance = displacement.norm()

            # Check if movement is within segment
            if proj_distance <= segment_length:
                # Update object position and velocity
                final_position += dir * proj_distance
                x_obj[obj_id, rid] = final_position
                v_obj[obj_id, rid] = current_velocity

                # Spring/damper force logic
                node1 = x_rope[rid, i_val]
                node2 = x_rope[rid, i_val+1]
                t = (final_position - node1).dot(node2 - node1) / segment_length**2
                closest_point_on_segment = node1 + t * (node2 - node1)
                force = spring_damper(final_position, closest_point_on_segment, v_obj[obj_id, rid], v_rope[rid, i_val], obj_spring, obj_damper, -1)
                v_obj[obj_id, rid] += force * dt / obj_mass

                # Splitting the force on the rope nodes
                f1 = (1 - t) * force
                f2 = t * force

                v_obj[obj_id, rid] += force * dt / obj_mass
                v_rope[rid, i_val] -= f1 * dt / rope_mass
                v_rope[rid, i_val+1] -= f2 * dt / rope_mass

                x_obj[obj_id, rid] += v_obj[obj_id, rid] * dt
                x_rope[rid, i_val] += v_rope[rid, i_val] * dt
                x_rope[rid, i_val+1] += v_rope[rid, i_val+1] * dt
                break
            else:
                final_position += dir * segment_length
                remaining_distance -= segment_length
                i_val += 1
                if i_val >= num_elements_total - 1:
                    break

# Dedicated function for the shackles until we fix the problem for shackles with sid = 0
@ti.func
def slide_along_rope_shackles(connections_obj_rope, v_obj, x_obj, v_rope, x_rope, obj_mass, rope_mass, obj_friction_coeff, obj_spring, obj_damper, collision_damper_coeff, dt, num_elements_total):
    """
    Models the interaction of an object sliding along a rope.

    """
    for rid, n, obj_id in connections_obj_rope:
        if obj_id == 0:
            continue
        i = connections_obj_rope[rid, n, obj_id]
        i_val = int(i[0])

        current_velocity = v_obj[n, obj_id, rid]
        remaining_distance = current_velocity.norm() * dt
        final_position = x_obj[n, obj_id, rid]

        while remaining_distance > 0:
            # Compute possible movement directions along the rope
            dir_i1 = (x_rope[rid, i_val+1] - x_rope[rid, i_val]).normalized()
            dir_i2 = (x_rope[rid, i_val] - x_rope[rid, i_val+1]).normalized()

            proj_dir_i1 = current_velocity.dot(dir_i1)
            proj_dir_i2 = current_velocity.dot(dir_i2)
            proj_dir = 0.0
            segment_length = (x_rope[rid, i_val+1] - x_rope[rid, i_val]).norm()
            dir = ti.Vector([0.0, 0.0, 0.0])

            # Logic to determine direction
            if proj_dir_i1 > proj_dir_i2:
                dir = dir_i1
                proj_dir = proj_dir_i1
            elif proj_dir_i2 > proj_dir_i1:
                dir = dir_i2
                proj_dir = proj_dir_i2

            # Friction
            friction_force = -obj_friction_coeff * current_velocity
            friction_acceleration = friction_force / obj_mass
            current_velocity += friction_acceleration * dt

            # Avoiding the object from moving back due to very low velocities
            if current_velocity.norm() < 1e-6:
                break

            # Compute how far the object would move in this iteration
            proj_distance = proj_dir * dt

            # Object collision handling logic
            displacement = x_obj[n, obj_id+1, rid] - x_obj[n, obj_id, rid]
            distance = displacement.norm()

            # Check if movement is within segment
            if proj_distance <= segment_length:
                overlap = proj_distance - distance
                # Handle overlaps/collisions
                if overlap > 0:
                    v1 = v_obj[n, obj_id, rid]
                    v2 = v_obj[n, obj_id+1, rid]
                    delta_v = (v1 - v2).dot(dir)  # Relative velocity along the collision direction

                    alpha = overlap / 2
                    damping_force = collision_damper_coeff * delta_v

                    v_obj[n, obj_id, rid] = v1 - alpha + damping_force
                    v_obj[n, obj_id+1, rid] = v2 + alpha - damping_force

                    correction1 = alpha * dir
                    correction2 = alpha * dir

                    x_obj[n, obj_id, rid] -= correction1
                    x_obj[n, obj_id+1, rid] += correction2
                    break
                else:
                    # Update object position and velocity
                    final_position += dir * proj_distance
                    x_obj[n, obj_id, rid] = final_position
                    v_obj[n, obj_id, rid] = current_velocity

                    # Spring/damper force logic
                    node1 = x_rope[rid, i_val]
                    node2 = x_rope[rid, i_val+1]
                    t = (final_position - node1).dot(node2 - node1) / segment_length**2
                    closest_point_on_segment = node1 + t * (node2 - node1)
                    force = spring_damper(final_position, closest_point_on_segment, v_obj[n, obj_id, rid], v_rope[rid, i_val], obj_spring, obj_damper, -1)
                    v_obj[n, obj_id, rid] += force * dt / obj_mass

                    # Splitting the force on the rope nodes
                    f1 = (1 - t) * force
                    f2 = t * force

                    v_obj[n, obj_id, rid] += force * dt / obj_mass
                    v_rope[rid, i_val] -= f1 * dt / rope_mass
                    v_rope[rid, i_val+1] -= f2 * dt / rope_mass

                    x_obj[n, obj_id, rid] += v_obj[n, obj_id, rid] * dt
                    x_rope[rid, i_val] += v_rope[rid, i_val] * dt
                    x_rope[rid, i_val+1] += v_rope[rid, i_val+1] * dt
                    break
            else:
                overlap = segment_length - distance
                # Handle overlaps/collisions
                if overlap > 0:
                    v1 = v_obj[n, obj_id, rid]
                    v2 = v_obj[n, obj_id+1, rid]
                    delta_v = (v1 - v2).dot(dir)  # Relative velocity along the collision direction

                    alpha = overlap / 2
                    damping_force = collision_damper_coeff * delta_v

                    v_obj[n, obj_id, rid] = v1 - alpha + damping_force
                    v_obj[n, obj_id+1, rid] = v2 + alpha - damping_force

                    correction1 = alpha * dir
                    correction2 = alpha * dir

                    x_obj[n, obj_id, rid] -= correction1
                    x_obj[n, obj_id+1, rid] += correction2
                    break
                else:
                    final_position += dir * segment_length
                    remaining_distance -= segment_length
                    i_val += 1
                if i_val >= num_elements_total - 1:
                    break