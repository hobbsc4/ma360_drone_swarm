import numpy as np

def environmental_model ( control_function ):
    """
    For a given control function (sensor data -> acceleration values), run an entire model pass.
    
    This means:
        - Initialize the model with a pre-defined initial state.
        - Move the model one step forwards.
            - Move all drones according to their velocities.
            - Update all drones' accelerations.
            - Update all drones' velocities.
            
            - Update target's orientation.
                - Move at a predefined angular rate toward the nearest drone.
                - If target is capable of pointing at drone in the current time-step,
                    update target orientation to point at drone.
            
            - Find all drones in target's weapon range AND angular range.
                - Kill nearest one.
            
            - If the target is within a drone's weapon range 
                AND the drone is allowed to fire, fire at the target
                (target has some number of hits required to take it out).

            - If a drone has fired, do not allow it to fire again.
                
        - Repeat this process until one of the following is true:
            - All drones are dead.
            - All targets are dead.
            - All drones have fired.
            - Number of time-steps has exceeded pre-determined limit
        
        - Return the number of drones left and the number of targets left.
    """
    
    # domain is centered at the origin. ex: domain_x = 10 => x ranges from -5 to 5
    
    TIME_STEP_SIZE = 1     # size of each time-step, in seconds
    DOMAIN_X       = 1000  # approximate width of the simulated domain
    DOMAIN_Y       = 1000  # approximate height of the simulated domain
    MAX_TIMESTEPS  = 500   # if the model has not reached another stop condition in this many time steps, it stops
    
    N_DRONES          = 100  # number of drones in the simulation
    DRONES_FIRE_RANGE = 30   # range from which drones can hit the target
    DRONE_MAX_ACCELERATION = 20  # maximum acceleration of a given drone
        
    TARGET_X             = np.array([ -1, 0,  1 ]) * DOMAIN_X / 2  # x-coordinate of each target
    TARGET_Y             = np.array([ -1, 1, -1 ]) * DOMAIN_Y / 2  # y-coordinate of each target
    N_TARGETS            = len(TARGET_X)                           # number of targets in the simulation
    TARGET_FIRE_RANGE    = 100                                     # range from which target can hit a drone
    TARGET_ANGULAR_SPEED = 20 * np.pi / 180                        # maximum angular speed of target's weapon (in radians/second)
    TARGET_HEALTH        = 10                                      # number of times each target must be hit to be disabled
    
    # initialize target health for each target
    target_health = TARGET_HEALTH * np.ones(N_TARGETS)
    
    # initialize target positions
    target_positions = np.hstack(
        (
            TARGET_X.reshape(-1, 1),
            TARGET_Y.reshape(-1, 1)
        )
    )
    
    # initialize target orientations
    target_orientations = np.zeros( (N_TARGETS) )  # angle (in radians) between target's weapon and the x-axis
    
    # initialize drone positions
    drone_positions_x = np.linspace(0, DOMAIN_X, N_DRONES) - DOMAIN_X / 2
    drone_positions_y = np.ones_like(drone_positions_x) * DOMAIN_Y / 2
    
    drone_positions = np.hstack(
        (
            drone_positions_x.reshape(-1, 1),
            drone_positions_y.reshape(-1, 1)
        )
    )
    
    # initialize drone velocities
    drone_velocities = np.zeros_like(drone_positions)  # drones begin as stationary objects
    
    # initialize masks and time counter variable
    drones_alive_mask  = np.ones( (N_DRONES,      1) )  # indices of drones which have not yet been killed
    drones_armed_mask  = np.ones( (N_DRONES,      1) )  # indices of drones which have not yet fired
    targets_alive_mask = np.ones( (len(TARGET_X), 1) )  # indices of targets which have not yet been killed
    current_time_step  = 0
    
    # model loop
    while (
        targets_alive_mask.sum() > 0  # stop if all targets are dead
        and
        drones_alive_mask.sum() > 0   # stop if all drones are dead
        and 
        drones_armed_mask.sum() > 0   # stop if drones can no longer fire
        and
        current_time_step <= MAX_TIMESTEPS  # stop if model runs too long
    ):
        # update positions
        drone_positions += drone_velocities * TIME_STEP_SIZE
        
        # get sensor data
        sensor_data = np.vstack(
            (
                drone_positions.reshape(-1,1),     # turn drone positions into a column vector
                target_positions.reshape(-1,1),    # turn target positions into a column vector
                target_orientations.reshape(-1,1)  # turn target orientations into a column vector
            )
        )
        
        # pass sensor data to control function to get acceleration
        drone_accelerations = control_function(sensor_data)
        
        # update velocities
        drone_velocities += drone_accelerations * TIME_STEP_SIZE
        
        drone_velocities * (drones_alive_mask != 1) = np.zeros_like( drone_velocities * (drones_alive_mask != 1) )  # dead drones don't move
        
        # update target orientations
        for i in range(N_TARGETS):
            if targets_alive_mask[i] == 0: continue  # dead targets can't move or fire
            
            current_target_position = target_positions[i].reshape(1,2)  # position of the current target in the loop
            vector_to_target        = drone_positions * drones_alive_mask - current_target_position  # vector from the target to each drone
            distance_to_target = np.sqrt(
                vector_to_target[:,0] ** 2
                +
                vector_to_target[:,1] ** 2 
            )  # Euclidean distance between each drone and the current target
            
            min_idx = distance_to_target.index(
                distance_to_target.min()
            )  # index of closest drone to target
            
            closest_drone_position = (drone_positions * drones_alive_mask)[min_idx, :]
            
            closest_drone_angle = np.arctan2(
                closest_drone_position[1] - current_target_position[1],  # y distance between drone and target
                closest_drone_position[0] - current_target_position[0]   # x distance between drone and target
            )
            
            angular_distance = closest_drone_angle - target_orientations[i]
            
            turning_direction = 0 if angular_distance == 0 else (
                angular_distance / np.abs(angular_distance)
            )  # -1: drone is clockwise from target, 1: drone is counterclockwise from target, 0: drone is in front of target
            
            # update target orientation
            if np.abs(angular_distance) < TARGET_ANGULAR_SPEED * TIME_STEP_SIZE:
                target_orientations[i] = closest_drone_angle
            else:
                target_orientations[i] += turning_direction * TARGET_ANGULAR_SPEED * TIME_STEP_SIZE
            
            # get all drones in range
            if distance_to_target[min_idx] < TARGET_FIRE_RANGE:
                drones_alive_mask[min_idx] = 0  # kill drone
                drones_armed_mask[min_idx] = 0  # obviously, if the drone is dead, it can't fire
            
        for i in range(N_DRONES):
            if drones_armed_mask[i] == 0: continue  # dead drones don't fire at targets

            current_drone_position = drone_positions[i].reshape(1,2)            # position of the current drone
            vector_to_drone        = target_positions - current_drone_position  # vector from each target to the current drone
            distance_to_drone      = np.sqrt(
                vector_to_target[:,0] ** 2
                +
                vector_to_target[:,1] ** 2 
            )  # Euclidean distance between each drone and the current target

            min_idx = distance_to_drone.index(
                distance_to_drone.min()
            )  # index of closest target to drone

            if distance_to_drone[min_idx] > DRONES_FIRE_RANGE: continue  # don't fire at target if target is out of range

            target_health[min_idx]    -= 1  # hit the target
            if target_health[min_idx] == 0: targets_alive_mask[min_idx] = 0  # kill target if it has no more health left
            drones_armed_mask[i]       = 0  # drone can no longer fire

        current_time_step += 1  # increment time step counter
    
    num_drones_left  = drones_alive_mask.sum()
    num_targets_left = targets_alive_mask.sum()
    return num_drones_left, num_targets_left

print(environmental_model( lambda x: np.ones( (100,2) ) ))