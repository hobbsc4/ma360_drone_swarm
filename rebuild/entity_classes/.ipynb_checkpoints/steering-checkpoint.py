import mesa
import numpy as np

from .target import Target

def boids(drone):
    # Initialize steering vectors
    alignment_steering  = np.zeros(2)
    cohesion_steering   = np.zeros(2)
    separation_steering = np.zeros(2)
    
    # For cohesion calculations
    center_of_mass = np.zeros(2)
    
    # For separation calculations
    avg_vector = np.zeros(2)
    
    # List of all agents - MAY CONTAIN TARGET
    neighbors = drone.model.domain.get_neighbors(drone.pos, drone.vis_radius, False)
    
    n_neighbors = 0  # Used to count neighbors
    for neighbor in neighbors:
        if isinstance(neighbor, Target): continue  # Skip target
        alignment_steering += neighbor.velocity       # Alignment
        center_of_mass     += np.array(neighbor.pos)  # Cohesion
        
        # Separation
        distance = np.linalg.norm(
            np.array(neighbor.pos)
            -
            np.array(drone.pos)
        )
        
        drone_to_neighbor  = np.array(drone.pos) - np.array(neighbor.pos)
        drone_to_neighbor /= distance
        avg_vector        += drone_to_neighbor
        
        n_neighbors += 1  # Increment n_neighbors
        
    if n_neighbors > 0 and np.linalg.norm(alignment_steering) != 0:
        
        # Alignment
        alignment_steering /= n_neighbors
        alignment_steering *= drone.max_velocity / np.linalg.norm(alignment_steering)
        alignment_steering -= drone.velocity
        
        # Cohesion
        center_of_mass /= n_neighbors
        com_direction   = center_of_mass - np.array(drone.pos)
        
        distance_to_com = np.linalg.norm(com_direction)
        if distance_to_com > 0:
            com_direction *= drone.max_velocity / distance_to_com
            
        cohesion_steering = com_direction - drone.velocity
        
        # Separation
        avg_vector /= n_neighbors
        
        if np.linalg.norm(avg_vector) > 0:
            avg_vector *= drone.max_velocity / np.linalg.norm(avg_vector)
        
        separation_steering = avg_vector - drone.velocity
            
    return alignment_steering, cohesion_steering, separation_steering

def avoid_edges(drone):
    margin_x = drone.model.domain.width  * .075
    margin_y = drone.model.domain.height * .075
    
    desired_velocity = np.zeros(2)
    steering = np.zeros(2)

    x, y = drone.pos
    
    influences = 0
    
    # Left boundary
    if x < drone.vis_radius and x < margin_x:
        desired_velocity[0] += 1 / x
        influences += 1
    
    # Right boundary
    if x > (drone.model.domain.width - drone.vis_radius) and x > drone.model.domain.width - margin_x:
        desired_velocity[0] -= 1 / (drone.model.domain.width - x)
        influences += 1
    
    # Bottom boundary
    if y < drone.vis_radius and y < margin_y:
        desired_velocity[1] += 1 / y
        influences += 1
    
    # Top boundary
    if y > (drone.model.domain.height - drone.vis_radius) and y > drone.model.domain.width - margin_y:
        desired_velocity[1] -= 1 / (drone.model.domain.height - y)
        influences += 1
    
    if influences > 0:
        desired_velocity /= influences
        desired_velocity *= drone.max_velocity / np.linalg.norm(desired_velocity)
        
        steering = desired_velocity - drone.velocity
    
    return steering

def seek_target(drone):
    steering = np.zeros(2)
    desired_velocity = np.zeros(2)
    
    target = drone.get_nearest_target()
    if not target: return steering  # do not steer if not targets in sight
    
    direction = np.array(target.pos) - np.array(drone.pos)
    
    desired_velocity = direction / np.linalg.norm(direction) * drone.max_velocity
    
    steering = desired_velocity - drone.velocity
    return steering
    
    
    
    