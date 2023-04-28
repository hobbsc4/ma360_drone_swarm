import mesa
import numpy as np

from entity_classes.drone  import Drone
from entity_classes.target import Target
from config import options

class SwarmModel(mesa.Model):
    def __init__(self, num_drones=options["num_drones"]):
        
        # Initialize options dict. as parameter of the model
        self.options = options
        
        # Set up the grid and schedule
        self.domain = mesa.space.ContinuousSpace(
            x_max=self.options["domain_width"], 
            y_max=self.options["domain_height"], 
            torus=True
        )
        self.schedule = mesa.time.RandomActivation(self)
        
        # Initialize some parameters
        self.initial_num_drones  = num_drones
        self.initial_num_targets = self.options["num_targets"]
        
        self.drone_cost   = self.options["drone_cost"]
        self.target_value = self.options["target_cost"]
        
        self.dt = 1  # size of each model time-step, in seconds
        
        # Variables to be used for counting
        self.current_num_drones       = self.initial_num_drones
        self.current_num_armed_drones = self.initial_num_drones 
        self.current_num_targets      = self.initial_num_targets
        self.current_id               = 0  # unique id of the next agent to be initialized
        
        drone_x = np.random.uniform( 
            low=self.domain.width * (1/2 - .2), 
            high=self.domain.width * (1/2 + .2), 
            size=self.initial_num_drones 
        )
        
        drone_y = np.random.uniform( 
            low= self.domain.height * (5/8), 
            high=self.domain.height * (7/8), 
            size=self.initial_num_drones 
        )
        
        # Initialize Drones
        for i in range(self.initial_num_drones):
            drone            = Drone(self.current_id, self)
            self.current_id += 1
            self.schedule.add(drone)
            
            x = drone_x[i]
            y = drone_y[i]
            
            self.domain.place_agent(drone, (x, y))
            
        self.target_y    = self.domain.height * 1/10
        target_x_margin  = self.domain.width / (self.initial_num_targets + 1)
        
        self.target_x = np.linspace(
            target_x_margin,
            self.domain.width - target_x_margin,
            self.initial_num_targets
        )
        
        # Initialize Targets
        for i in range(self.initial_num_targets):
            target           = Target(self.current_id, self)
            self.current_id += 1
            self.schedule.add(target)
            
            x = self.target_x[i]
            y = self.target_y
            
            self.domain.place_agent(target, (x, y))           
        
    def step(self):
        """
        Advance the model by one step.
        """
        self.schedule.step()
    
    def run(self, num_drones=options["num_drones"]):
        self.__init__(num_drones)
        while (
            self.current_num_drones > 0  # end if targets take out drones
            and
            self.current_num_targets > 0  # end if drones take out targets
            and
            self.current_num_armed_drones > 0  # end if all drones have spent their ammunition
        ):
            self.step()
            
        num_drones_killed  = self.initial_num_drones - self.current_num_drones
        num_targets_killed = self.initial_num_targets - self.current_num_targets
        
        utility = (
            num_targets_killed
            *
            self.target_value
            -
            num_drones_killed
            *
            self.drone_cost
        )
        return utility