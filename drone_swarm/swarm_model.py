import mesa

from math import floor
from entity_classes.drone import Drone
from entity_classes.target import Target

class SwarmModel(mesa.Model):
    """
    A Mesa model for simulating a drone swarm.

    Attributes
    ----------
    initial_n_drones : int
        The number of drones initially in the swarm.
    DRONE_DIAMETER : float
        The diameter of each drone, in meters.
    target_vis_radius : float
        The visibility radius of the target, in meters.
    drone_vis_radius : float
        The visibility radius of each drone, in meters.
    drone_weapon_radius : float
        The weapon radius of each drone, in meters.
    dt : int
        The amount of time each time step represents, in seconds.
    current_id : int
        The ID of the next agent to be added to the model.
    schedule : mesa.time.RandomActivation
        The scheduler that governs the agents' actions.
    domain : mesa.space.ContinuousSpace
        The space in which the agents move.
    datacollector : mesa.DataCollector
        The data collector that collects data on the agents.

    Methods
    -------
    __init__(n_drones, domain_width, domain_height, drone_diameter, target_vis_radius, drone_vis_radius, drone_weapon_radius)
        Initializes the model with the specified parameters.
    step()
        Advances the model by one time step.
    """
    
    def __init__(
        self,
        n_drones:               int,
        domain_width:           float,
        domain_height:          float,
        target_vis_radius:      float,
        target_weapon_range:    float,
        drone_vis_radius:       float,
        drone_weapon_radius:    float,
        drone_max_accuracy:     float,
        drone_max_velocity:     float,
        drone_max_acceleration: float,
        weapon_angular_range:   float,
        fire_cooldown:          float,
        omega_max:              float,
        dt:                     float
    ):
        """
        Initializes the swarm model with the specified parameters.

        Parameters
        ----------
        n_drones : int
            The number of drones in the swarm.
        domain_width : float
            The width of the simulation domain, in meters.
        domain_height : float
            The height of the simulation domain, in meters.
        drone_diameter : float
            The diameter of each drone, in meters.
        target_vis_radius : float
            The visibility radius of the target, in meters.
        drone_vis_radius : float
            The visibility radius of each drone, in meters.
        drone_weapon_radius : float
            The weapon radius of each drone, in meters.
        
        Raises
        ------
        ValueError
            If the simulation domain is too small for the number and size of the drones.
        """
        
        self.running = True
        
        self.initial_n_drones = n_drones
        self.initial_n_targets = 1
        
        self.target_vis_radius   = target_vis_radius
        self.target_weapon_range = target_weapon_range
        self.drone_vis_radius    = drone_vis_radius
        self.drone_weapon_radius = drone_weapon_radius
        self.drone_max_accuracy  = drone_max_accuracy
        self.drone_max_velocity  = drone_max_velocity
        self.drone_max_acceleration = drone_max_acceleration
        
        self.drone_value  = 5000 # USD / drone
        self.target_value = 1e6
        
        self.weapon_angular_range = weapon_angular_range
        self.fire_cooldown        = fire_cooldown
        self.omega_max            = omega_max
        
        self.n_drones_returned = 0  # counts how many drones have made it back to the boundary after firing
        
        # Amount of time each time step represents
        self.dt = dt # second
        
        # Next available unique id for an agent
        self.current_id = 0
        
        # Initialize scheduler
        # Note: a random scheduler may not be beneficial for our purposes. It is used 
        # here for simplicity, but we should consider (if necessary), using a different 
        # scheduler which can be found at:
        #      https://mesa.readthedocs.io/en/stable/apis/time.html
        self.schedule = mesa.time.RandomActivation(self)
        
        self.domain = mesa.space.ContinuousSpace(
            x_max=domain_width,
            y_max=domain_height,
            torus=True   # Set to true for testing. Make false later. We do not want the space to "wrap around."
                         # Drones should not be able to teleport from left to right/top to bottom/etc. as
                         # it is not realistic.
        )
        
        for _ in range(self.initial_n_drones):
            drone = Drone(
                self.current_id,
                self,
                self.drone_vis_radius, 
                self.drone_weapon_radius,
                self.drone_max_accuracy,
                self.drone_max_velocity,
                self.drone_max_acceleration)  # Initialize each drone
            
            self.current_id += 1 #  Increment current_id
            
            self.schedule.add(drone)  # Add drone to the scheduler
            
            # Add the drone to a particular position - randomly for now
            x = self.random.randrange(self.domain.width * (.5 - .25), self.domain.width * (.5 + .25))
            y = self.random.randrange(self.domain.height * 7/8, self.domain.height)
            self.domain.place_agent(drone, (x, y))
        
        # Initialize target        
        target = Target(self.current_id, self, self.target_vis_radius, self.target_weapon_range, self.weapon_angular_range, self.fire_cooldown, self.omega_max)
        self.schedule.add(target)
        self.current_id += 1 # Increment current_id
        
        self.target = target # store target here as well
        
        # Put it in the middle for now
        x = self.domain.width / 2
        y = self.domain.height / 2
        self.domain.place_agent(target, (x, y))
        
        # Use this to collect data, exports as a pandas dataframe
        self.datacollector = mesa.DataCollector(
            agent_reporters={"Position": "pos"}
        )
        return
    
    def get_num_drones(self):
        agents = self.schedule.agents
        n_drones = 0
        for agent in agents:
            if not isinstance(agent, Target): n_drones += 1
        return n_drones
    
    def get_num_targets(self):
        agents = self.schedule.agents
        n_targets = 0
        for agent in agents:
            if isinstance(agent, Target): n_targets += 1
        return n_targets
    
    def get_cost(self):
        n_drones  = self.get_num_drones()
        n_targets = self.get_num_targets()
        drone_value = self.drone_value
        target_value = self.target_value
        return (
            ( self.initial_n_drones - n_drones )
            *
            drone_value
            -
            ( self.initial_n_targets - n_targets )
            *
            target_value
        )
    
    def end(self):
        self.running = False
        print(f"Drones Left: {self.get_num_drones()}\nTargets Left: {self.get_num_targets()}.\nCost = {self.get_cost()}")
        return
        
    def step(self):
        """
        Advance the model by one time step.
        """
        if self.get_num_drones() <= 0:
            self.end()
        elif self.get_num_targets() <= 0:
            self.end()
        else:
            self.schedule.step()  # Advance the scheduler

"""
Docstrings generated by ChatGPT.
"""