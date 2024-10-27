# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script launches the crowd navigation environment and optionally plots or saves observations.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import open3d as o3d

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="My environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--plot_obs_pos", type=int, default=-1, help="Plots the positions of the obstacles in the given frame."
)
parser.add_argument("--plot_pc_env", type=int, default=-1, help="Plots the lidar pointcloud of the given robot.")
parser.add_argument(
    "--plot_2d",
    action="store_true",
    default=False,
    help="When plotting the point coud, plot 2d teacher pointcloud instead of 3d.",
)
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Saves the point clouds and metadata to disk. Path has to be specified in the code. Each data point is saved as a ~1 MB json file.",
)

parser.add_argument(
    "--time_between_saves",
    type=int,
    default=1,
    help="Time in seconds to save the point clouds. Default all 1 seconds",
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
import os

# from icecream import ic
# from icecream import ic

from omni.isaac.lab.envs import ManagerBasedRLEnv


from crowd_navigation_mt.agent_config.anymal_d.navigation_env_cfg import (
    AnymalDCrowdNavigationTeacherEnvCfg,
    AnymalDCrowdNavigationRecordingEnvCfg,
)

import datetime
import json

import matplotlib.pyplot as plt

current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

POINTCLOUD_SAVE_PATH = f"/home/rafael/Projects/PLR/data/{current_time}"

SAVE_POINTCLOUDS = args_cli.save
PLOT_2D_PC = args_cli.plot_2d  # Plot 2D pointclouds instead of 3D
PLOT_POSITIONS = args_cli.plot_obs_pos != -1
PLOT_POINTCLOUD = args_cli.plot_pc_env != -1
##
# plotting
##


def initialize_plot():
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    sc = ax.scatter([], [])  # Initialize an empty scatter plot
    ax.scatter(0, 0, color="red")  # Red point at the origin
    ax.grid(True)  # Enable grid
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    env_nr = args_cli.plot_obs_pos if args_cli.plot_obs_pos != -1 else args_cli.plot_pc_env
    plt.title(f"Dynamic 2D Points Plotting, env={env_nr}")

    # Set fixed axes limits to make the plot square
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)

    ax.set_aspect("equal")

    plt.show()
    return fig, ax, sc


def update_plot(fig, ax, sc, tensor):
    # Ensure the tensor is in the shape (n_points, 2)
    if tensor.ndim != 2 or tensor.shape[1] != 2:
        raise ValueError("Tensor must be of shape (n_points, 2)")

    x, y = tensor[:, 0], tensor[:, 1]  # Extract x and y coordinates
    sc.set_offsets(torch.stack((x, y), dim=1))  # Update the scatter plot with new data

    # Redraw the current figure with updates
    plt.draw()
    plt.pause(0.01)


def initialize_point_cloud():
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Placeholder for initial random point cloud
    points_data = torch.rand(100, 3) * 100  # Random 3D points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_data.numpy())

    # Use matplotlib's colormap
    cmap = plt.get_cmap("coolwarm")
    z_norm = (points_data[:, 2] - torch.min(points_data[:, 2])) / (
        torch.max(points_data[:, 2]) - torch.min(points_data[:, 2])
    )
    colors = cmap(z_norm.numpy())[:, :3]  # Get RGB from RGBA
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis.add_geometry(pcd)

    # Add smaller coordinate axes to the plot
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    vis.add_geometry(axis_frame)

    # Set view parameters
    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    ctr.set_front([0, 0, 1])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.25)  # Adjust this value to fit the 30x30 view, may need fine-tuning

    return vis, pcd


def update_point_cloud(vis, pcd, new_point_tensor):
    if new_point_tensor.ndim != 2 or new_point_tensor.shape[1] != 3:
        raise ValueError("Tensor must be of shape (N_points, 3)")

    # Update the points in the point cloud object
    new_points = new_point_tensor.numpy()
    pcd.points = o3d.utility.Vector3dVector(new_points)

    # Update colors based on z-values using matplotlib's colormap
    cmap = plt.get_cmap("coolwarm")
    z_norm = (new_point_tensor[:, 2] - torch.min(new_point_tensor[:, 2])) / (
        torch.max(new_point_tensor[:, 2]) - torch.min(new_point_tensor[:, 2])
    )
    colors = cmap(z_norm.numpy())[:, :3]  # Get RGB from RGBA
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()


def save_to_json(data, file_name: str):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.join(POINTCLOUD_SAVE_PATH), exist_ok=True)
    path = os.path.join(POINTCLOUD_SAVE_PATH, file_name)

    # Check if the file already exists
    if os.path.exists(path):
        raise FileExistsError(f"The file {path} already exists.")

    # Write the data to a JSON file
    with open(path, "w") as file:
        json.dump(data, file)


def main():
    """Main function."""
    # create environment configuration
    env_cfg = AnymalDCrowdNavigationRecordingEnvCfg()
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    if args_cli.cpu:
        env_cfg.sim.device = "cpu"
        env_cfg.sim.use_gpu_pipeline = False
        env_cfg.sim.physx.use_gpu = False

    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    if PLOT_POSITIONS:
        fig, ax, sc = initialize_plot()

    if PLOT_POINTCLOUD:
        if PLOT_2D_PC:
            fig2, ax2, sc2 = initialize_plot()
        else:
            vis, pcd = initialize_point_cloud()

    size = (args_cli.num_envs**0.5 + 1) * env_cfg.scene.env_spacing

    # initial actions
    # actions = torch.rand_like(env.action_manager.get_term("velocity_cmd").raw_actions)
    # actions = torch.ones_like(env.action_manager.get_term("velocity_command").raw_actions) * 0
    # actions = torch.ones_like(env.action_manager.get_term("velocity_command").raw_actions) * 0
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    # actions = torch.ones_like(env.action_manager.get_term("velocity_cmd").raw_actions) * 0
    actions[:, 0] -= 0.5
    actions[:, 0] += 1
    # positions = (torch.rand_like(env.action_manager.get_term("obstacle_bot_positions").raw_actions) - 0.5) * size
    # actions = torch.cat((robot_action, positions), dim=1)

    # dataset initialization
    if SAVE_POINTCLOUDS:
        lidar_meshes_regex = env.scene.sensors["lidar"].cfg.mesh_prim_paths
        lidar_meshes = ["No_mesh"]

        for item in lidar_meshes_regex:
            if ".*" in item:
                # If ".*" is present, replace it with each number from 0 to N-1
                for i in range(env_cfg.scene.num_envs):
                    # Replace ".*" with the current number and add to the new list
                    new_item = item.replace(".*", str(i))
                    lidar_meshes.append(new_item)
            else:
                # If ".*" is not present, just add the original item
                lidar_meshes.append(item)
        save_to_json(lidar_meshes, "lidar_meshes.json")

    count = 0
    count_global = 0
    count_save = 0
    count_resets = 0
    reset_time = 10  # s
    update_after_seconds = 2
    update_time = update_after_seconds
    save_every_seconds = args_cli.time_between_saves
    # simulate physics

    while simulation_app.is_running():
        with torch.inference_mode():
            # # reset
            if count % int(reset_time / env.step_dt) == 0:
                count = 0
                count_resets += 1
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                # target_positions = torch.randn_like(env.action_manager.action)
                # actions[:, 2] *= 0

            print(f"[INFO]: t = {round(count * env.step_dt, 2)}")

            # step the environment
            obs, rew, terminated, truncated, info = env.step(actions)

            if PLOT_POSITIONS:
                p_x = obs["positions"]["positions"][args_cli.plot_obs_pos][:, 0]
                p_y = obs["positions"]["positions"][args_cli.plot_obs_pos][:, 1]
                update_plot(fig, ax, sc, torch.stack((p_x, p_y), dim=1).cpu())

            if PLOT_POINTCLOUD:
                if not PLOT_2D_PC:
                    update_point_cloud(vis, pcd, obs["lidar_raw"][args_cli.plot_pc_env].cpu())
                else:
                    update_plot(fig2, ax2, sc2, obs["lidar_2d"]["lidar_full"][args_cli.plot_pc_env][:, :2].cpu())

            if SAVE_POINTCLOUDS:
                # save_pointcloud(
                #     pointclouds=obs["lidar_privileged"]["lidar"],
                #     iter=count,
                #     reset_count=count_resets,
                #     segmentation_mask=obs["lidar_privileged"]["segmentation"],
                #     obstacle_positions=obs["obstacle_positions"],
                # )

                if count_global % int(save_every_seconds / env.step_dt) == 0 and count_global > 0:
                    print(f"[INFO]: Saving pointclouds at t = {round(count_global * env.step_dt, 2)} s. ...")

                    for env_id in range(env.num_envs):
                        data = {
                            "pc_full": obs["lidar_raw"][env_id].to(torch.float16).cpu().tolist(),
                            "label_pc_2d_full": obs["lidar_2d"]["lidar_full"][env_id][:, :2]
                            .to(torch.float16)
                            .cpu()
                            .cpu()
                            .tolist(),
                            "label_pc_2d_distances": obs["lidar_2d"]["lidar_label_distances"][env_id]
                            .to(torch.float16)
                            .squeeze()
                            .cpu()
                            .tolist(),
                            "label_pc_2d_mesh_velocities": obs["lidar_2d"]["lidar_label_mesh_velocities"][env_id]
                            .to(torch.float16)
                            .cpu()
                            .tolist(),
                            "label_pc_2d_segmentation": obs["lidar_2d"]["segmentation"][env_id]
                            .squeeze()
                            .cpu()
                            .tolist(),
                        }

                        save_to_json(data, f"env_{env_id}_pc_{count_save}.json")
                    count_save += 1

            # ic(robot_action.shape)
            # ic(obs["lidar"].shape)
            # ic(actions.shape)

            # update counter
            count += 1
            count_global += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
