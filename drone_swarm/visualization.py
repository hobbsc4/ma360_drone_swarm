from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer

from swarm_model import SwarmModel


def swarm_portrayal(agent):
    """
    This function determines how the agents are drawn on the canvas.
    """
    if agent is None:
        return

    portrayal = {}

    if isinstance(agent, Drone):
        portrayal["Shape"] = "circle"
        portrayal["r"] = 0.5
        portrayal["Layer"] = 0
        portrayal["Filled"] = "true"
        portrayal["Color"] = "red"
    elif isinstance(agent, Target):
        portrayal["Shape"] = "rect"
        portrayal["w"] = 1
        portrayal["h"] = 1
        portrayal["Layer"] = 0
        portrayal["Filled"] = "true"
        portrayal["Color"] = "green"

    return portrayal


# Create the canvas and set the dimensions based on the domain size
grid_width = 20
grid_height = 20
canvas_width = 500
canvas_height = 500

grid = CanvasGrid(swarm_portrayal, grid_width, grid_height, canvas_width, canvas_height)

# Create the model and pass it to the server
model_params = {
    "n_drones": 10,
    "domain_width": 100,
    "domain_height": 100,
    "drone_diameter": 1,
    "target_vis_radius": 5,
    "drone_vis_radius": 10,
    "drone_weapon_radius": 2
}

model = SwarmModel(**model_params)
server = ModularServer(SwarmModel, [grid], "Swarm Model", model_params)

server.port = 8521
server.launch()
