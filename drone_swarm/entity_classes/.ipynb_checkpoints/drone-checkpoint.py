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
    
    def __init__(self, unique_id, model, vis_range, weapon_range, max_accuracy, max_velocity, max_acceleration):
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
        self.all_states = ["flocking", "retreating", "returned"]
        self.state      = self.all_states[0] # drones flocking initially
        
        self.vis_range        = vis_range
        self.weapon_range     = weapon_range
        self.max_velocity     = max_velocity  # meters / second
        self.max_acceleration = max_acceleration  # m / s^2
        self.max_accuracy     = max_accuracy # must be between 0 and 1
        
        # Initialize kinematic parameters
        self.velocity     = (np.random.rand(2) - .5) * self.max_velocity
        self.acceleration = np.random.rand(2) - .5  
        
        # Initialize steering weights
        self.weights  = np.array([ 
            1,     # alignment weight
            .7,    # cohesion weight
            1.05,  # separation weight
            1.2,     # edge avoidance weight
            1,     # target seeking weight
        ])
    
    def update_position(self):
        """
        Updates the position of the drone based on its current velocity and the time step of the model.
        """
        next_position = np.array(self.pos) + self.model.dt * self.velocity
        
        next_x, next_y = next_position
        
        if next_x > self.model.domain.width or next_x < 0 or next_y > self.model.domain.height or next_y < 0:
            self.state = self.all_states[2] # set drone state to "returned"
            return
        
        # move drone
        self.model.domain.place_agent(self, tuple(next_position))
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
        self.acceleration = (
            1
            /
            self.weights.sum()
            *
            (
                self.weights[0] * alignment_steering 
                + 
                self. weights[1] * cohesion_steering 
                + 
                self.weights[2] * separation_steering
                +
                self.weights[3] * edge_avoidance_steering
                +
                self.weights[4] * target_seeking_steering
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
    
    def die(self):
        self.model.domain.remove_agent(self)
        self.model.schedule.remove(self)
        self.model.current_id -= 1
        return
    
    def fire(self):
        if self.state == self.all_states[1]: return # do not fire if retreating
        target = self.model.target
        xt, yt = target.pos
        
        xs, ys = self.pos
        
        distance = np.linalg.norm([ xt - xs, yt - ys ])
        
        if distance > self.weapon_range: return # do not fire if target is out of range
        
        # Probability of getting a fatal shot at the target decreases linearly with distance (should probably find a better model)
        probability = self.max_accuracy * (1 - distance/self.weapon_range)
        
        target.get_hit(probability) # hit the target
        
        # Set drones to fly away from the target, towards the edge once they have fired
        self.weights[3] *= -1 
        self.weights[4] *= -1
        self.state = self.all_states[1] # Set drones to "retreating" state
        return
                
    def step(self):
        """
        Executes a single step of the drone agent behavior.
        Updates the position, acceleration, and velocity of the drone, in that order.
        """
        self.update_position()
        
        # if the drone has returned, remove it from the model and increment the returned counter
        if self.state == self.all_states[2]:
            self.die()
            self.model.n_drones_returned += 1
            return
        
        self.update_acceleration()
        self.update_velocity()
        self.fire()
        
"""
Docstrings generated by ChatGPT.
"""