import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from typing import TYPE_CHECKING

# if TYPE_CHECKING:
from crowd_navigation_mt.terrains import SemanticTerrainImporterCfg

def create_semantic_map(device, size, cfg: SemanticTerrainImporterCfg | None ):
    """
    This function currently just creates a static map representation with labels as follows:
    sidewalk: 0
    Crosswalk: 1
    Park: 2
    Street: 3
    House / Static Obstacle: 4
    """

    # Define map size and resolution
    width = size[0]  # in m
    height = size[1]  # in m
    if cfg is None:
        resolution = 0.1
    else:
        resolution = cfg.semantic_terrain_resolution

    grid_size = (int(width / resolution), int(height / resolution))  # convert to grid cells

    # Initialize the map grid with default values 
    grid_map = torch.zeros((grid_size), dtype=int, device=device) 

    # Define regions
    # Sidewalk everywhere initialized as zeros

    # Street
    grid_map[:, int(7 / resolution):int(13 / resolution) + 1] = 3
    grid_map[int(4  / resolution):int(10 / resolution) + 1, :] = 3
    grid_map[int(18 / resolution):int(24 / resolution) + 1, :] = 3

    # Crosswalk
    grid_map[int(2 / resolution) :int(4  / resolution) + 1, int(7  / resolution):int(13 / resolution) + 1] = 1
    grid_map[int(10 / resolution):int(12 / resolution) + 1, int(7  / resolution):int(13 / resolution) + 1] = 1
    grid_map[int(16 / resolution):int(18 / resolution) + 1, int(7  / resolution):int(13 / resolution) + 1] = 1
    grid_map[int(24 / resolution):int(26 / resolution) + 1, int(7  / resolution):int(13 / resolution) + 1] = 1
    grid_map[int(4 / resolution) :int(10 / resolution) + 1, int(5  / resolution):int(7  / resolution) + 1] = 1
    grid_map[int(18 / resolution):int(24 / resolution) + 1, int(5  / resolution):int(7  / resolution) + 1] = 1
    grid_map[int(4 / resolution) :int(10 / resolution) + 1, int(13 / resolution):int(15 / resolution) + 1] = 1
    grid_map[int(18 / resolution):int(24 / resolution) + 1, int(13 / resolution):int(15 / resolution) + 1] = 1

    # Park
    grid_map[int(12 / resolution):int(16 / resolution) + 1, 0:int(2 / resolution) + 1] = 2
    grid_map[int(26 / resolution):, int(15 / resolution):] = 2     

    # House/Static Obstacles
    grid_map[int(12 / resolution):int(16 / resolution) + 1, int(15 / resolution):] = 4

    if cfg.debug_plot:
        plot_semantic_terrain(grid_map=grid_map, name="Grid_Map")

    return grid_map


def plot_semantic_terrain(grid_map: torch.tensor, name: str):
    # with open("tensor.csv", "w") as file:
    #     for row in grid_map:
    #         csv_row = ",".join(map(str, row.tolist()))
    #         file.write(f"{csv_row}\n")

    # # Define colormap 
    colors = ["gray", "blue", "green", "yellow", "red", "purple"]
    cmap = ListedColormap(colors)

    # get num unique classes in the grid map
    num_classes = torch.unique(grid_map).cpu().numpy()

    # Symbolic representation of the map
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid_map.cpu().T, cmap=cmap, origin="lower", vmin=0, vmax=len(num_classes))

    # Add a legend
    legend_labels = ["Sidewalks", "Crosswalks", "Park", "Street", "House/Static Obstacle", "Dynamic Obstacle"]
    colors = [cmap(i / 6) for i in range(6)]

    # Define legend patches based on the colormap
    legend_patches = [
        mpatches.Patch(color=color, label=label)
        for color, label in zip(colors, legend_labels)
    ]
    ax.legend(handles=legend_patches,  loc="upper right")

    ax.set_title(f"{name}_Representation")
    ax.set_xlabel("Grid Cell X")
    ax.set_ylabel("Grid Cell Y")

    # Save the figure as an image
    output_path = f"crowd_navigation_mt/exts/crowd_navigation_mt/crowd_navigation_mt/terrains/elevation_map/output/{name}.png"
    plt.savefig(output_path)
    print(f"Map saved to {output_path}")
    plt.close()


# To debug the function
# create_semantic_map("cuda:0", (29.0, 20.0))
