"""
Run to watch plot update in real time.
"""

import mesa
import numpy as np
from swarm_model import SwarmModel
from entity_classes.target import Target
from entity_classes.drone import Drone

from IPython import display
import matplotlib.pyplot as plt

def main():
    model = SwarmModel([0.5,0.5,2,0.5,0.5], 90)

    # create a figure and axis objects
    fig, ax = plt.subplots()
    fig.show()

    # set axis limits
    ax.set_xlim([0, model.domain.width])
    ax.set_ylim([0, model.domain.height])
    ax.set_aspect('equal')

    # Define a function to update the plot
    def update_plot(ax):
        agents = model.schedule.agents
        drones = []
        targets = []
        for agent in agents:
            if isinstance(agent, Drone): drones.append(agent)
            if isinstance(agent, Target): targets.append(agent)
        
        # clear the axis
        ax.clear()

        # set axis limits
        ax.set_xlim([0, model.domain.width])
        ax.set_ylim([0, model.domain.height])
        
        for target in targets:
            arrow_angle = target.direction
            arrow_dx = 100 * np.cos(arrow_angle)
            arrow_dy = 100 * np.sin(arrow_angle)
            
            target_x, target_y = target.pos
            
            ax.plot(target_x, target_y, 'ro', markersize=2)
            ax.arrow(target_x, target_y, arrow_dx, arrow_dy, color='red', linewidth=.5)
            
        # plot all the drones as blue circles
        for drone in drones:
            drone_x, drone_y = drone.pos
            ax.plot(drone_x, drone_y, 'bo')
        plt.title(model.schedule.time)

    # Update plot in real-time
    while (
            model.current_num_drones > 0  # end if targets take out drones
            and
            model.current_num_targets > 0  # end if drones take out targets
            and
            model.current_num_armed_drones > 0  # end if all drones have spent their ammunition
            and
            model.schedule.time < 400
        ):
        model.step()
        update_plot(ax)
        plt.pause(.01) # Pause for a short time

if __name__ == "__main__":
    main()