## This file contains the class definition of the airplane agent. Comments were revised using ChatGPT

import mesa

class Airplane(mesa.Agent):
    """ An agent representing an airplane. """

    def __init__(self, unique_id, model):
        """
        Initialize a new instance of the Airplane class.
        Args:
            unique_id (int): Unique identifier for the agent.
            model (mesa.Model): The model object the agent belongs to.
        """
        super().__init__(unique_id, model)

    def move(self):
        """
        Move the agent according to some rule.
        """
        x, y = self.pos
        x += self.random.random() - 0.5
        y += self.random.random() - 0.5
        self.model.spatial_domain.move_agent(self, (x, y))

    def step(self):
        """
        Define the agent's behavior for each time step.
        """
        self.move()
