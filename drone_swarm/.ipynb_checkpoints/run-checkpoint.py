"""
Run to watch plot update in real time.
"""

import mesa
import numpy as np
from swarm_model import SwarmModel
from entity_classes.target import Target

from IPython import display
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
    model = SwarmModel(
        n_drones               = 25,    # number of drones in the model
        domain_width           = 1000,  # width of the spatial domain, in meters
        domain_height          = 1000,  # height of the spatial domain, in meters
        target_vis_radius      = 1000,  # target's visibility range, in meters
        target_weapon_range    = 400,   # effective weapon range of target, in meters
        drone_vis_radius       = 500,   # drone's visibility radius, in meters
        drone_weapon_radius    = 100,   # effective weapon radius of drone, in meters
        drone_max_accuracy     = .9,    # probability between 0 and 1
        drone_max_velocity     = 27,    # meters / second
        drone_max_acceleration = 20,                # meters / second^2
        weapon_angular_range   = 10 * np.pi / 180,  # radians
        fire_cooldown          = 2,                 # seconds
        omega_max              = 45 * np.pi / 180,  # max. angular velocity of target's weapon, in radians / second
        dt                     = 1                  # model time-step size, in seconds
    )

    target_x, target_y = model.target.pos

    # target orientation parameters
    arrow_half_len = model.target_weapon_range / 6
    arrow_angle    = model.target.direction
    arrow_start_x  = target_x
    arrow_start_y  = target_y
    arrow_dx       = arrow_half_len * np.cos(arrow_angle)
    arrow_dy       = arrow_half_len * np.sin(arrow_angle)

    # create a figure and axis objects
    fig, ax = plt.subplots()

    # set axis limits
    ax.set_xlim([0, model.domain.width])
    ax.set_ylim([0, model.domain.height])

    # plot the target's vis. radius
    theta = np.linspace(0, 2*np.pi)
    vrx, vry = model.target_weapon_range * np.array([ np.cos(theta), np.sin(theta) ])
    vrx += target_x
    vry += target_y
    ax.plot(vrx, vry, 'r', linewidth=.5)

    # plot the target's orientation
    ax.arrow(arrow_start_x, arrow_start_y, arrow_dx, arrow_dy, color='r', linewidth=.5)
    ax.set_aspect('equal')

    # plot all the drones as blue circles
    for drone in model.schedule.agents:
        if isinstance(drone, Target): continue
        drone_x, drone_y = drone.pos
        ax.plot(drone_x, drone_y, 'bo')

    # Define a function to update the plot
    def update_plot(ax):
        # clear the axis
        ax.clear()

        # set axis limits
        ax.set_xlim([0, model.domain.width])
        ax.set_ylim([0, model.domain.height])

        target_x, target_y = model.target.pos

        # target orientation parameters
        arrow_angle       = model.target.direction
        arrow_start_x     = target_x
        arrow_start_y     = target_y
        arrow_dx          = arrow_half_len * np.cos(arrow_angle)
        arrow_dy          = arrow_half_len * np.sin(arrow_angle)

        # plot the target as a red circle
        target_x, target_y = model.target.pos
        ax.plot(target_x, target_y, 'ro', markersize=2)

        # plot the target's vis. radius
        ax.plot(vrx, vry, 'r', linewidth=.5)

        # plot the target's orientation
        ax.arrow(arrow_start_x, arrow_start_y, arrow_dx, arrow_dy, color='red', linewidth=.5)

        # plot all the drones as blue circles
        for drone in model.schedule.agents:
            if isinstance(drone, Target): continue
            drone_x, drone_y = drone.pos
            ax.plot(drone_x, drone_y, 'bo')

    # Update plot in real-time
    while model.running:
        model.step()
        if model.get_num_targets() > 0: update_plot(ax) # Update plot
        plt.pause(0.01) # Pause for a short time

if __name__ == "__main__":
    main()