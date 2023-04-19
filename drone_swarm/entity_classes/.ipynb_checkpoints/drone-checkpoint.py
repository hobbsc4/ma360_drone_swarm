import mesa
import numpy as np

from .target import Target
from .steering import boids, avoid_edges, seek_target

class Drone(mesa.Agent):
    """
    Represents a drone agent in a simulation.
    
    Attributes:
    -----------
    unique_id: int
        A unique identifier for the drone.
    model: mesa.Model
        The model containing the drone.
    diameter: float
        The diameter of the drone in meters.
    vis_range: float
        The visibility range of the drone in meters.
    weapon_range: float
        The weapon range of the drone in meters.
    max_velocity: float
        The maximum velocity of the drone in meters per second.
    max_acceleration: float
        The maximum acceleration of the drone in meters per second squared.
    velocity: numpy.ndarray
        The current velocity of the drone in meters per second.
    acceleration: numpy.ndarray
        The current acceleration of the drone in meters per second squared.
    
    Methods:
    --------
    update_position():
        Updates the position of the drone.
    update_acceleration():
        Calculates and updates the acceleration of the drone based on its current state.
    update_velocity():
        Updates the velocity of the drone based on its current acceleration.
    step():
        Advances the drone forward one time step.
    """
    
    def __init__(self, unique_id, model, diameter, vis_range, weapon_range):
        """
        Initializes a drone agent.
        
        Parameters:
        -----------
        unique_id: int
            A unique identifier for the drone.
        model: mesa.Model
            The model containing the drone.
        diameter: float
            The diameter of the drone in meters.
        vis_range: float
            The visibility range of the drone in meters.
        weapon_range: float
            The weapon range of the drone in meters.
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
        """
        Updates the position of the drone based on its current velocity and the time step of the model.
        """        
        self.model.domain.place_agent(
            self,
            tuple(
                np.array(self.pos) + self.model.dt * self.velocity
            )
        )
        return
        
    def update_acceleration(self):
        """
        Calculates and updates the acceleration of the drone based on its current state and the weights assigned to different
        types of steering behaviors.
        """
        
        # Calculate boids steering/acceleration vectors
        alignment_steering, cohesion_steering, separation_steering = boids(self)
        edge_avoidance_steering = avoid_edges(self)
        target_seeking_steering = seek_target(self)
        
        # Calculate total steering/acceleration vector
        weights  = np.array([ 
            1,  # alignment weight
            1,  # cohesion weight
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
        """
        Updates the velocity of the drone according to the acceleration and time step.
        If the norm of the new velocity exceeds the maximum velocity, it is clamped to that value.
        """
        self.velocity += self.acceleration * self.model.dt
        
        # Clamp velocity
        if np.linalg.norm(self.velocity) > self.max_velocity:
            self.velocity *= self.max_velocity / np.linalg.norm(self.velocity)
        return
    
    def step(self):
        """
        Executes a single step of the drone agent behavior.
        Updates the position, acceleration, and velocity of the drone, in that order.
        """
        self.update_position()
        self.update_acceleration()
        self.update_velocity()
        
"""
Docstrings generated by ChatGPT.
"""