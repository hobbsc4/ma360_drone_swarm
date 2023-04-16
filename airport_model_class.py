## This file contains the class definition of the model. Comments were revised using ChatGPT

import mesa

# Import agent classes
from airplane_agent_class import Airplane

class AirportModel(mesa.Model):
    """
    A model that contains all agents.
    """

    def __init__(self, n_airplanes, max_width, max_height):
        """
        Initialize a new instance of the AirportModel class.
        Args:
            n_airplanes (int): Number of airplanes in the model.
            max_width (int): Width of spatial component of the model.
            max_height (int): Height of spatial component of the model.
        """
        
        self.n_airplanes = n_airplanes  # Set the number of airplanes
        
        # Initialize scheduler
        self.schedule = mesa.time.RandomActivation(self)
        
        # Initialize spatial component of the model
        self.spatial_domain = mesa.space.ContinuousSpace(
            x_max=max_width,
            y_max=max_height,
            torus=True  # Whether the space is toroidal.
        )
        
        # Create airplanes
        for i in range(self.n_airplanes):
            plane = Airplane(i, self)  # Initialize each airplane
            self.schedule.add(plane)  # Add airplane to the scheduler
            
            # Add the agent to a particular position - randomly for now
            x = self.random.randrange(self.spatial_domain.width)
            y = self.random.randrange(self.spatial_domain.height)
            self.spatial_domain.place_agent(plane, (x, y))
            
        # Note: [MODEL].schedule.agents holds all agent objects
        
        # Use this to collect data, exports as a pandas dataframe
        self.datacollector = mesa.DataCollector(
            agent_reporters={"Position": "pos"}
        )

    def step(self):
        """
        Advance the model by one step.
        """
        self.datacollector.collect(self)  # Collect data
        self.schedule.step()  # Advance the scheduler
