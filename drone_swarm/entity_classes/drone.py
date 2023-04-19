import mesa
import numpy as np

from .target import Target
from .steering import boids, avoid_edges, seek_target

class Drone(mesa.Agent):
    
    def __init__(self, unique_id, model, diameter, vis_range, weapon_range):
        """
        Initializes a drone agent.
        """
        super().__init__(unique_id, model)
        self.diameter         = diameter  # Each drone is assumed to be a circle of a given diameter (meters)
        self.vis_range        = vis_range
        self.weapon_range     = weapon_range
        self.max_velocity     = 50  # meters / second
        self.max_acceleration = 50  # m / s^2
        
        # Initialize kinematic parameters
        self.velocity     = (np.random.rand(2) - .5) * self.max_velocity
        self.acceleration = np.random.rand(2) - .5  
    
    def update_position(self):
        # Move the drone
        self.model.domain.place_agent(
            self,
            tuple(
                np.array(self.pos) + self.model.dt * self.velocity
            )
        )
        return
        
    def update_acceleration(self):
        # Calculate boids steering/acceleration vectors
        alignment_steering, cohesion_steering, separation_steering = boids(self)
        edge_avoidance_steering = avoid_edges(self)
        target_seeking_steering = seek_target(self)
        
        # Calculate total steering/acceleration vector
        weights  = np.array([ 
            1,   # alignment weight
            .95,  # cohesion weight
            1.05,  # separation weight
            1,  # edge avoidance weight
            1,  # target seeking weight
        ])
        
        self.acceleration = (
            1
            /
            weights.sum()
            *
            (
                weights[0] * alignment_steering 
                + 
                weights[1] * cohesion_steering 
                + 
                weights[2] * separation_steering
                +
                weights[3] * edge_avoidance_steering
                +
                weights[4] * target_seeking_steering
            )
        )
        
        # Clamp acceleration
        if np.linalg.norm(self.acceleration) > self.max_acceleration:
            self.acceleration *= self.max_acceleration / np.linalg.norm(self.acceleration)
        return
    
    def update_velocity(self):
        # Update velocity according to v_i = v_(i-1) + a * dt
        self.velocity += self.acceleration * self.model.dt
        
        # Clamp velocity
        if np.linalg.norm(self.velocity) > self.max_velocity:
            self.velocity *= self.max_velocity / np.linalg.norm(self.velocity)
        return
    
    def step(self):
        """
        Describes drone behavior with each schedule.step()
        """
        self.update_position()
        self.update_acceleration()
        self.update_velocity()
        
        