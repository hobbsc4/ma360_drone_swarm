import mesa
import numpy as np

class Target(mesa.Agent):
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        
        # Initialize target parameters
        self.vis_radius = self.model.options["target_vis_radius"]
        self.weapon_range = self.model.options["target_weapon_range"]
        self.fire_cooldown = self.model.options["target_fire_cooldown"]
        self.omega_max = self.model.options["target_max_turn_rate"]
        
        # Initialize kinematic parameters
        self.direction = np.random.uniform(0, 2 * np.pi)
        
        # Initialize state variables
        self.all_states = ["alive", "dead"]
        self.state = self.all_states[0]
        self.time_until_fire = self.fire_cooldown
    
    def get_nearest_drone(self):
        neighbors = self.model.domain.get_neighbors(self.pos, self.vis_radius, False)
        if not neighbors: return None # if no drones are visible, return None
        
        drone_neighbors = []
        for neighbor in neighbors:
            if isinstance(neighbor, Target): continue  # don't consider other targets
            drone_neighbors.append(neighbor)
        
        if not drone_neighbors: return None  # if no drones are visible, return None
    
        distances = [
            np.linalg.norm(
                np.array(neighbor.pos) 
                - 
                np.array(self.pos)
            ) 
            for neighbor in drone_neighbors
        ]
        min_distance = min(distances)
        min_distance_idx = distances.index(min_distance)
        
        nearest_drone = drone_neighbors[min_distance_idx]
        return nearest_drone
    
    def move(self):
        if self.state == self.all_states[1]: return  # don't move if dead
    
        nearest_drone = self.get_nearest_drone()
        if not nearest_drone: return
        
        xs, ys = self.pos
        
        xn, yn = nearest_drone.pos
        neighbor_angle = np.arctan2(yn - ys, xn - xs)
        
        theta_diff = neighbor_angle - self.direction
                
        desired_omega = theta_diff / self.model.dt
        
        if desired_omega <= self.omega_max: 
            self.direction = neighbor_angle
        else:
            self.direction += self.omega_max * self.model.dt
    
    def fire(self):
        if self.time_until_fire > 0:
            self.time_until_fire -= self.model.dt
            return
        
        nearest_drone = self.get_nearest_drone()
        if not nearest_drone: return  # don't fire if no drones in range
        
        xs, ys = self.pos
        xn, yn = nearest_drone.pos
        nearest_drone_angle = np.arctan2(yn - ys, xn - xs)
        
        drone_distance = np.sqrt(
            ( xn - xs ) ** 2
            +
            ( yn - ys ) ** 2
        )
        
        if drone_distance > self.weapon_range: return  # don't fire if drone is out of range
        
        if np.abs(nearest_drone_angle - self.direction) > 1e-2: return  # don't fire if weapon pointed the wrong way
        
        nearest_drone.die()  # "kill" the drone
        
        # reset weapon cooldown
        self.time_until_fire = self.fire_cooldown
    
    def get_hit(self, probability):
        random_number = np.random.rand()
        
        if random_number <= probability:
            self.state = self.all_states[1]
            self.model.current_num_targets -= 1
    
    def step(self):
        self.move()
        self.fire()