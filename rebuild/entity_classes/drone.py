import mesa
import numpy as np

from .steering import boids, avoid_edges, seek_target

class Drone(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        # Drone states
        self.all_states = ["flocking", "retreating"]
        self.state      = self.all_states[0]
        
        # Initialize drone parameters
        self.vis_radius       = self.model.options["drone_vis_radius"]
        self.weapon_radius    = self.model.options["drone_weapon_radius"]
        self.max_velocity     = self.model.options["drone_max_velocity"]
        self.max_acceleration = self.model.options["drone_max_acceleration"]
        self.accuracy         = self.model.options["drone_accuracy"]
        
        self.steering_weights = np.array([
            1,  # alignment
            1,  # cohesion
            1,  # separation
            1,  # edge avoidance
            1,  # target seeking
        ]);
        
        # Initialize kinematic parameters
        initial_direction = np.random.uniform(-np.pi, 0)
        self.velocity = np.array([
            np.cos(initial_direction),
            np.sin(initial_direction)
        ]) * self.max_velocity
        
        self.acceleration = np.array([
            np.cos(initial_direction),
            np.sin(initial_direction)
        ]) * self.max_acceleration
    
    def update_position(self):
        next_position = np.array(self.pos) + self.model.dt * self.velocity
        self.model.domain.move_agent(self, next_position)
    
    def update_acceleration(self):
        
        # Calculate boids steering/acceleration vectors
        alignment_steering, cohesion_steering, separation_steering = boids(self)
        edge_avoidance_steering = avoid_edges(self)
        target_seeking_steering = seek_target(self)
        
        # Calculate total steering/acceleration vector        
        self.acceleration = (
            1
            /
            self.steering_weights.sum()
            *
            (
                self.steering_weights[0] * alignment_steering 
                + 
                self.steering_weights[1] * cohesion_steering 
                + 
                self.steering_weights[2] * separation_steering
                +
                self.steering_weights[3] * edge_avoidance_steering
                +
                self.steering_weights[4] * target_seeking_steering
            )
        )
        
        # Clamp acceleration
        if np.linalg.norm(self.acceleration) > self.max_acceleration:
            self.acceleration *= self.max_acceleration / np.linalg.norm(self.acceleration)
            
    def update_velocity(self):
        self.velocity += self.acceleration * self.model.dt
        
        # Clamp velocity
        if np.linalg.norm(self.velocity) > self.max_velocity:
            self.velocity *= self.max_velocity / np.linalg.norm(self.velocity)
    
    def die(self):
        self.model.domain.remove_agent(self)
        self.model.schedule.remove(self)
        self.model.current_num_drones -= 1
        
        if not self.state == self.all_states[1]: self.model.current_num_armed_drones -= 1
        self.model.current_id -= 1
    
    def get_nearest_target(self):
        neighbors = self.model.domain.get_neighbors(
            self.pos, 
            self.vis_radius,
            False
        )
        if not neighbors: return None  # if no drones are visible, return None
    
        target_neighbors = []
        for neighbor in neighbors:
            if isinstance(neighbor, Drone): continue  # don't consider drones
            if neighbor.state == neighbor.all_states[1]: continue  # don't shoot at dead targets
            target_neighbors.append(neighbor)
        
        if not target_neighbors: return None  # if no targets are visible, return None
        
        distances = [
            np.linalg.norm(
                np.array(neighbor.pos)
                -
                np.array(self.pos)) 
            for neighbor in target_neighbors
        ]
        min_distance     = min(distances)
        min_distance_idx = distances.index(min_distance)
        
        nearest_target = target_neighbors[min_distance_idx]
        return nearest_target
    
    def fire(self):
        if self.state == self.all_states[1]: return  # don't fire if already fired
    
        nearest_target = self.get_nearest_target()
        if not nearest_target: return  # don't fire if no targets in sight
    
        target_distance = np.sqrt(
            ( nearest_target.pos[0] - self.pos[0] ) ** 2
            +
            ( nearest_target.pos[1] - self.pos[1] ) ** 2
        )
        
        if target_distance > self.weapon_radius: return  # don't fire at out-of-range targets
        
        nearest_target.get_hit(self.accuracy)
        self.state = self.all_states[1]  # set drone to "retreating"
        self.model.current_num_armed_drones -= 1
        
    def step(self):
        self.update_position()
        self.update_acceleration()
        self.update_velocity()
        self.fire()