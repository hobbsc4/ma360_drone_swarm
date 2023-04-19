import mesa

from math import floor
from entity_classes.drone import Drone
from entity_classes.target import Target

class SwarmModel(mesa.Model):
    """
    A Mesa model for simulating a drone swarm.

    Attributes
    ----------
    N_DRONES : int
        The number of drones in the swarm.
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
        n_drones: int,
        domain_width: float, 
        domain_height: float, 
        drone_diameter: float, 
        target_vis_radius: float,
        drone_vis_radius: float,
        drone_weapon_radius: float
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
        
        # There is almost certainly a better way to do this. However this is meant to ensure that 
        # the domain is of a reasonable size for the number and size of drones specified
        if domain_width / n_drones < drone_diameter or domain_height / n_drones < drone_diameter:
            raise ValueError(
                f"""
                Domain is too small. Must be at least {drone_diameter * n_drones} by {drone_diameter * n_drones} for the specified number
                and size of drones.
                """)
        
        self.DRONE_DIAMETER = drone_diameter
        self.N_DRONES       = n_drones
        
        self.target_vis_radius   = target_vis_radius
        self.drone_vis_radius    = drone_vis_radius
        self.drone_weapon_radius = drone_weapon_radius
        
        # Amount of time each time step represents
        self.dt = 1 # second
        
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
        
        for _ in range(self.N_DRONES):
            drone = Drone(
                self.current_id,
                self,
                self.DRONE_DIAMETER, 
                self.drone_vis_radius, 
                self.drone_weapon_radius)  # Initialize each drone
            
            self.current_id += 1 #  Increment current_id
            
            self.schedule.add(drone)  # Add drone to the scheduler
            
            # Add the drone to a particular position - randomly for now
            x = self.random.randrange(-self.domain.width, self.domain.width)
            y = self.random.randrange(self.domain.height * 3/4, self.domain.height)
            self.domain.place_agent(drone, (x, y))
        
        # Initialize target
        self.target_id = self.current_id  # Store target ID for use later
        
        target = Target(self.target_id, self, self.target_vis_radius)
        self.schedule.add(target)
        self.current_id += 1 # Increment current_id
        
        # Put it in the middle for now
        x = self.domain.width / 2
        y = self.domain.height / 2
        self.domain.place_agent(target, (x, y))
        
        # Use this to collect data, exports as a pandas dataframe
        self.datacollector = mesa.DataCollector(
            agent_reporters={"Position": "pos"}
        )
        
    def step(self):
        """
        Advance the model by one time step.
        """
        self.datacollector.collect(self)  # Collect data
        self.schedule.step()  # Advance the scheduler

"""
Docstrings generated by ChatGPT.
"""