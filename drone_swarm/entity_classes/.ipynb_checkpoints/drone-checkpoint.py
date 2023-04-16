import mesa
import numpy as np

from .target import Target

class Drone(mesa.Agent):
    
    def __init__(self, unique_id, model, initial_state, diameter, vis_range, weapon_range):
        """
        Initializes a drone agent.
        """
        super().__init__(unique_id, model)
        self.diameter     = diameter  # Each drone is assumed to be a circle of a given diameter (meters)
        self.state        = initial_state
        self.vis_range    = vis_range
        self.weapon_range = weapon_range
        self.max_velocity = 1  # meters / second
        
        self.search_direction = None  # Will be removed later. For now, stores search direction
        self.search_duration  = 0     # Another hacky, temporary variable. Used to store length of time drone has been searching
    
    def determine_goal_position(self):
        
        # For now, move randomly at full speed if the drones are in the search state
        if self.state == self.model.DRONE_STATES[0]:
            
            # Change direction every minute
            if self.search_duration % 60 == 0:
                self.search_direction = np.random.uniform(0, 2 * np.pi)
                
            direction  = np.array([ np.cos(self.search_direction), np.sin(self.search_direction) ])
            self_pos   = np.array(self.pos)
            position   = tuple( self_pos + direction * self.max_velocity * self.model.dt )
            
            # Increment search_duration. REMOVE LATER
            self.search_duration += 1
            return position
        
        # For now, move directly towards target at full speed if drones are in attack state
        if self.state == self.model.DRONE_STATES[1]:
            target_id  = self.model.target_id
            target_pos = np.array(self.model.schedule.agents[target_id].pos)
            self_pos   = np.array(self.pos)
            direction  = target_pos - self_pos
            direction /= np.linalg.norm(direction)
            position   = tuple( self_pos + direction * self.max_velocity * self.model.dt )
            return position
        
    def target_in_range(self):
        neighbors = self.model.domain.get_neighbors(
            self.pos,
            self.vis_range,
            include_center=False
        )
        
        if any( [isinstance(neighbor, Target) for neighbor in neighbors] ):
            return True
        
        return False
    
    def step(self):
        """
        Describes drone behavior with each schedule.step()
        """
        
        goal_position = self.determine_goal_position()  # Determine goal position
        self.model.domain.move_agent(self, goal_position)  # Move agent to goal position
        
        # If target is in range, 
        if self.target_in_range():
            for agent in self.model.schedule.agents:
                if isinstance(agent, Target): continue  # Don't modify target state
                
                agent.state = self.model.DRONE_STATES[1]  # Set drone states to attack