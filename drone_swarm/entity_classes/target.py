import mesa
import numpy as np

class Target(mesa.Agent):
    
    def __init__(self, unique_id, model, vis_radius, weapon_range, weapon_angular_range, fire_cooldown, omega_max):
        """
        Initializes a target agent.
        """
        super().__init__(unique_id, model)
        
        self.all_states = ["defending", "dead"]
        self.state = self.all_states[0]
        
        self.direction = np.random.uniform(0, 2*np.pi)
        
        # Initialize target's visibility range. Once a drone comes into
        # this range, the target will attempt to shoot it.
        self.vis_radius = vis_radius
        
        # Angular range in which weapon can hit drones
        self.weapon_angular_range = weapon_angular_range
        
        # Firing range of weapon
        self.weapon_range = weapon_range 
        
        # Fire rate of weapon. Used to clamp number of drones it can hit in a given time frame.
        # This can be changed later
        self.fire_cooldown = fire_cooldown  # (seconds / shot)
        
        # Time until weapon can fire again
        self.time_until_fire = self.fire_cooldown
        
        # Set maximum angular velocity of target (assumed 15 deg/sec for now)
        self.omega_max = omega_max
    
    def get_nearest_neighbor(self):
        neighbors = self.model.domain.get_neighbors(self.pos, self.vis_radius, False)
        if not neighbors: return None # if no drones are visible, return None
        
        distances        = [np.linalg.norm(np.array(neighbor.pos) - np.array(self.pos)) for neighbor in neighbors]
        min_distance     = min(distances)
        min_distance_idx = distances.index(min_distance)
        
        nearest_neighbor = neighbors[min_distance_idx]
        return nearest_neighbor
        
    def move(self):
        nearest_neighbor = self.get_nearest_neighbor()
        if not nearest_neighbor: return
        
        xs, ys = self.pos
        
        xn, yn = nearest_neighbor.pos
        neighbor_angle = np.arctan2(yn - ys, xn - xs)
        
        theta_diff = neighbor_angle - self.direction
                
        desired_omega = theta_diff / self.model.dt
        
        if desired_omega <= self.omega_max: 
            self.direction = neighbor_angle
        else:
            self.direction += self.omega_max * self.model.dt
        return
    
    def fire(self):
        if self.time_until_fire > 0:
            self.time_until_fire -= self.model.dt
            return
        
        nearest_neighbor = self.get_nearest_neighbor()
        if not nearest_neighbor: return
        
        nearest_neighbor_distance = np.linalg.norm(np.array(nearest_neighbor.pos) - np.array(self.pos))
        
        xs, ys = self.pos
        
        xn, yn = nearest_neighbor.pos
        nearest_neighbor_angle = np.arctan2(yn - ys, xn - xs)
        
        # don't fire if the drone is out of range
        if nearest_neighbor_distance > self.weapon_range: return
        
        # don't fire if the weapon is pointed the wrong way
        if np.abs(nearest_neighbor_angle - self.direction) > self.weapon_angular_range / 2: return 
        
        # "Kill" the drone
        nearest_neighbor.die()
        
        # Reset the weapon cooldown
        self.time_until_fire = self.fire_cooldown
        return
    
    def get_hit(self, probability):
        random_number = np.random.rand()
        
        if random_number <= probability:
            self.state = self.all_states[1] # set target state to "dead"
            self.model.end()
        return
    
    def step(self):
        """
        Describes target behavior with each schedule.step()
        """
        self.move()
        self.fire()