import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

def create_semantic_map(device, size, cfg = None):
    
    debug_plot = True

    # Define map size and resolution
    width = size[0]  # in m
    height = size[1]  # in m
    resolution = 0.1  # in m
    grid_size = (int(width / resolution), int(height / resolution))  # convert to grid cells

    # Initialize the map grid with default values 
    grid_map = torch.zeros((grid_size), dtype=int, device=device) 

    # Define regions
    # Sidewalk everywhere initialized as zeros

    # Street
    grid_map[70:131, :] = 3
    grid_map[:, 40:101] = 3
    grid_map[:, 180:241] = 3

    # Crosswalk
    grid_map[70:131, 20:41] = 1
    grid_map[70:131, 100:121] = 1
    grid_map[70:131, 160:181] = 1
    grid_map[70:131, 240:261] = 1
    grid_map[50:71, 40:101] = 1
    grid_map[50:71, 180:241] = 1
    grid_map[130:151, 40:101] = 1
    grid_map[130:151, 180:241] = 1

    # Park
    grid_map[0:21, 120:161] = 2
    grid_map[150:, 260:] = 2     

    # House/Static Obstacles
    grid_map[150:, 120:161] = 4  

    if debug_plot:

        with open("tensor.csv", "w") as file:
            for row in grid_map:
                csv_row = ",".join(map(str, row.tolist()))
                file.write(f"{csv_row}\n")

        # Define colormap and boundaries
        colors = ["gray", "blue", "green", "yellow", "red"]
        cmap = ListedColormap(colors)
        boundaries = [0, 1, 2, 3, 4, 5]  # Boundaries for the color mapping
        norm = BoundaryNorm(boundaries, cmap.N, clip=True)

        # Symbolic representation of the map
        fig, ax = plt.subplots(figsize=(8, 8))
        # cmap = plt.get_cmap("tab20")
        im = ax.imshow(grid_map.cpu(), cmap=cmap, origin="lower")

        # Add a legend
        legend_labels = ["Sidewalks", "Crosswalks", "Park", "Street", "House/Static Obstacle"]
        colors = [cmap(i / 5) for i in range(5)]
        handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[i], markersize=10) for i in range(5)]
        ax.legend(handles, legend_labels, loc="upper right")

        # Define legend patches based on the colormap
        legend_patches = [
            mpatches.Patch(color=color, label=label)
            for color, label in zip(colors, ["Sidewalk", "Crosswalk", "Park", "Street", "House/Static Obstacle"])
        ]
        ax.legend(handles=legend_patches,  loc="upper right")

        ax.set_title("Grid Map Representation")
        ax.set_xlabel("Grid Cell X")
        ax.set_ylabel("Grid Cell Y")

        # Save the figure as an image
        output_path = "crowd_navigation_mt/exts/crowd_navigation_mt/crowd_navigation_mt/terrains/elevation_map/debug/grid_map.png"
        plt.savefig(output_path)
        print(f"Map saved to {output_path}")


    return grid_map


# To debug the function
# create_semantic_map("cuda:0", (29.0, 20.0))
