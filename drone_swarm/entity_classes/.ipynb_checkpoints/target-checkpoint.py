import mesa
import numpy as np

class Target(mesa.Agent):
    
    def __init__(self, unique_id, model, vis_radius):
        """
        Initializes a target agent.
        """
        super().__init__(unique_id, model)
        
        self.direction = np.random.uniform(0, 2*np.pi)
        
        # Initialize target's visibility range. Once a drone comes into
        # this range, the target will attempt to shoot it.
        self.vis_radius = vis_radius
        
        # Angular range in which weapon can hit drones
        self.weapon_angular_range = 15 * np.pi / 180
        
        # Fire rate of weapon. Used to clamp number of drones it can hit in a given time frame.
        # This can be changed later
        self.fire_rate = 1 / 15  # one shot every 15 seconds
        
        # Set maximum angular velocity of target (assumed 3 deg/sec for now)
        self.omega_max = 15 * np.pi / 180
        
    def step(self):
        """
        Describes target behavior with each schedule.step()
        """
        self.direction += self.omega_max * self.model.dt
        
        