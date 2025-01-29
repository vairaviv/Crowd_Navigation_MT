from __future__ import annotations

import torch
import numpy as np
import math
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

from collections.abc import Sequence

from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from omni.isaac.lab.managers import ManagerTermBase, SceneEntityCfg
from omni.isaac.lab.sensors import RayCaster
from omni.isaac.lab.assets import Articulation
import omni.isaac.lab.utils.math as math_utils


import crowd_navigation_mt.mdp as mdp  # noqa: F401, F403
from crowd_navigation_mt.terrains import SemanticTerrainImporter
from .semantic_map_obs_cfg import SemanticMapObsCfg

from omni.isaac.lab.utils import math as math_utils


class SemanticMapObs(ManagerTermBase):

    def __init__(self, cfg: SemanticMapObsCfg, env: ManagerBasedRLEnv):
        """Initialize the lidar history term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        self.debug_plot = cfg.debug_plot
        self.plot_env_id = cfg.plot_env_id

        super().__init__(cfg, env)
        if isinstance(env.scene.terrain, SemanticTerrainImporter):
            self.grid_res = env.scene.terrain.cfg.semantic_terrain_resolution
            self.grid_tf_vec = env.scene.terrain.transform_vector / self.grid_res
            grid_size = torch.tensor(env.scene.terrain.cfg.terrain_generator.size).to(device=env.device)

            # pad the grid map with streets, in case it is close to the border
            self.grid_map_padded, self.pad_tf_vec = self._initialize_padded_grid_map()
            if self.debug_plot:
                from crowd_navigation_mt.terrains.elevation_map.semantic_height_map import plot_semantic_terrain
                plot_semantic_terrain(self.grid_map_padded, name="Grid_Map_Padded")


            # observation size
            self.obs_x = cfg.obs_range[0] / self.grid_res
            self.obs_y = cfg.obs_range[1] / self.grid_res

            # the lookup matrix for observation, doesnt need to be integeres as rotated anyways later
            self.grid_filter_coord = torch.stack(
                torch.meshgrid(
                    torch.arange(-self.obs_x / 2, self.obs_x / 2, device=self.device),
                    torch.arange(-self.obs_y / 2, self.obs_y / 2, device=self.device)
                    # torch.arange(-int(self.obs_x / 2), int(self.obs_x / 2), device=self.device),
                    # torch.arange(-int(self.obs_y / 2), int(self.obs_y / 2), device=self.device)
                ), dim=-1
            ).view(-1, 2)
            
            # create a one hot version of the grid map
            num_classes = torch.unique(self.grid_map_padded).size(0)
            self.grid_map_one_hot = torch.nn.functional.one_hot(self.grid_map_padded, num_classes)

            # the root asset for observation
            self.robot: Articulation = env.scene[cfg.asset_cfg.name]

            if cfg.obstacle_cfg is None:
                print("[INFO]: Dynamic Obstacles are !!!not!!! added to the semantic observation.")
                self.obstacles = None
            else:
                print("[INFO]: Dynamic Obstacles are added to the semantic observations.")
                self.obstacles = env.scene[cfg.obstacle_cfg.name]
                self.obstacles_pos_w = env.scene[cfg.obstacle_cfg.name].data.root_pos_w
                self.obstacles_vel_w = env.scene[cfg.obstacle_cfg.name].data.root_vel_w
                self.num_obstacles = None
                radius = cfg.obstacle_buffer_radius
                offsets = torch.arange(-int(radius / self.grid_res), int(radius / self.grid_res) + 1)
                dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
                distance_mask = (dx ** 2 + dy ** 2) <= (radius / self.grid_res)
                self.obstacle_buffer_filter = torch.stack([dx[distance_mask], dy[distance_mask]], dim=-1).to(device=self.device)
                self._create_dynamic_obstacle_layer()
                
        else:
            raise TypeError(f"Expected an instance of SemanticTerrainImporter, but got {type(env.scene.terrain)}")
        

        

    def __call__(self, *args, method) -> torch.Any:

        # method_name = method.get("method")

        method = getattr(self, method, None)

        if method is None or not callable(method):
            raise ValueError(f"Method '{method}' not found in the class.")
            
        return method()

    def reset(self, env_ids: Sequence[int] | None = None):
        pass
    
    def get_map(self):
        # TODO: @vairaviv remove before run
        plot_all_obs = False
        plot_semantic_map = True

        # handles cases befor the terrain is initialized
        if self._env._sim_step_counter  == 0:
            semantic_map = torch.zeros(
                self.num_envs, 
                int(self.obs_x), 
                int(self.obs_y), 
                self.grid_map_one_hot.shape[-1], 
                device=self.device
            ).flatten(1, -1)

            return semantic_map

        else:
            # initialize once the amount of dynamic obstacles, checking for height > 0, over ground plane as all
            # other obstacles are spawned below the ground plane
            if self.num_obstacles is None and self.obstacles:
                self.num_obstacles = int(torch.sum(self.obstacles_pos_w[:, 2] > 0.0, dim=0))

            if self.obstacles:  # False: #
                obstacles_pos_w = self.obstacles.data.root_pos_w[:self.num_obstacles, :2]
                self._update_dyn_obstacle_layer(obstacles_pos_w)
            
            robot_pos_w = self.robot.data.root_pos_w.squeeze(1)[:, :2]
            robot_ang_z = math_utils.axis_angle_from_quat(self.robot.data.root_quat_w)[:, 2]

            rotated_coord = self.grid_filter_coord @ self._get_rot_mat(robot_ang_z)  # [num_envs, (obs_range/grid_res)**2, 2]

            coord_w = rotated_coord + robot_pos_w.unsqueeze(1) / self.grid_res 

            # transform the world coordinates to idx for the padded grid map
            idx = (coord_w - self.pad_tf_vec - self.grid_tf_vec).round().int()

            semantic_map = self.grid_map_one_hot[
                idx[:, :, 0], idx[:, :, 1]
            ].view(self.num_envs, int(self.obs_x) * int(self.obs_y), self.grid_map_one_hot.shape[-1])

            # Debug visualizations
            if self.debug_plot:
                # self.plot_obs_shape(idx)
                # self.plot_1_env_obs_shape(idx, 0)
                robot_pos_idx = (robot_pos_w / self.grid_res - self.pad_tf_vec - self.grid_tf_vec).round().int()
                if plot_all_obs:
                    for env_id in range(self.num_envs):
                        self.plot_1_env_obs_semantics(idx, semantic_map, robot_pos_idx.cpu(), robot_ang_z.cpu(), env_id)
                else:
                    self.plot_1_env_obs_semantics(idx, semantic_map, robot_pos_idx.cpu(), robot_ang_z.cpu(), self.plot_env_id)

                if plot_semantic_map:
                    from crowd_navigation_mt.terrains.elevation_map.semantic_height_map import plot_semantic_terrain
                    dyn_sem_map = torch.argmax(self.grid_map_one_hot, dim=-1).to(device="cpu")
                    plot_semantic_terrain(dyn_sem_map, name=f"Semantic_Map_Step_{self._env._sim_step_counter}")
                
            # for easier handling later: [n_envs, num_classes * (obs_rang/grid_res)²], thats why it is permuted here
            semantic_map = semantic_map.permute(0, 2, 1).flatten(1, -1)

            return semantic_map

    """
    Helpers
    """

    def _get_rot_mat(self, angle: torch.tensor):

        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)

        rotation_matrices = torch.stack(
            [
                torch.stack([cos_angle, -sin_angle], dim=1),
                torch.stack([sin_angle, cos_angle], dim=1),
            ],
            dim=2
        ).to(device=self.device)

        return rotation_matrices

    def _initialize_padded_grid_map(self):
        padding_value = 4  # for static obstacle
        if isinstance(self.cfg, SemanticMapObsCfg):
            padding_x = math.ceil(math.sqrt(2) * self.cfg.obs_range[0] / (2 * self.grid_res))
            padding_y = math.ceil(math.sqrt(2) * self.cfg.obs_range[1] / (2 * self.grid_res))
            if isinstance(self._env.scene.terrain, SemanticTerrainImporter):
                padded_grid_map = torch.nn.functional.pad(
                    self._env.scene.terrain.grid_map,
                    pad=(padding_x, padding_x, padding_y, padding_y),
                    mode="constant",
                    value=padding_value
                ).to(device=self.device)
                pad_transform_vec = torch.tensor([-padding_x, -padding_y]).to(device=self.device)

                return padded_grid_map, pad_transform_vec
            else:
                raise TypeError(f"Expected an instance of SemanticTerrainImporter, but got {type(self._env.scene.terrain)}")
        else:
            raise TypeError(f"Expected an instance of SemanticMapObsCfg, but got {type(self.cfg)}")
    
    def _create_dynamic_obstacle_layer(self):
        
        layer = torch.zeros_like(self.grid_map_padded)
        self.grid_map_one_hot = torch.cat((self.grid_map_one_hot, layer.unsqueeze(2)), dim=-1)
    
    def _update_dyn_obstacle_layer(self, obstacle_pos_w: torch.tensor):
        pos_idx = (obstacle_pos_w / self.grid_res - self.grid_tf_vec - self.pad_tf_vec).round().int()

        # grid_idx = self.obstacle_buffer_filter + pos_idx.unsqueeze(1)
        grid_idx = (
            self.obstacle_buffer_filter.to(device=self.device) + pos_idx.unsqueeze(1)
        ).view(self.num_obstacles * self.obstacle_buffer_filter.shape[0], 2)
        layer = torch.zeros_like(self.grid_map_padded)
        layer[grid_idx[:, 0], grid_idx[:, 1]] = 2
        # this assumes that the last layer in the one hot map is the one for dynamic obstacles
        self.grid_map_one_hot[:, :, -1] = layer

    ##
    # Debug Helpers & Plotters
    ##

    def plot_1_env_obs_shape(self, idx: torch.tensor, env_id: int = 0):

        res = self.grid_res

        plt.figure(figsize=(self.grid_map_padded.shape[0] * res, self.grid_map_padded.shape[1] * res))
        plt.title("Observation filter visualized")


        points = (idx[env_id] * res).cpu()
        plt.scatter(points[:, 0], points[:, 1], c="blue", s=1)
        plt.xlim(0, self.grid_map_padded.shape[0] * res)
        plt.ylim(0, self.grid_map_padded.shape[1] * res)
        # plt.axis("off")
        plt.xlabel("X Coordinate", fontsize=6)
        plt.ylabel("Y Coordinate", fontsize=6)

        plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

        # Save the figure as an image
        output_path = f"crowd_navigation_mt/exts/crowd_navigation_mt/crowd_navigation_mt/mdp/observations/output/obs_env_{env_id}_step_{self._env._sim_step_counter}.png"
        plt.savefig(output_path)
        print(f"Map saved to {output_path}")

        plt.close()

    def plot_1_env_obs_semantics(
            self, idx: torch.tensor, 
            semantic_map: torch.tensor, 
            robot_pos_idx: torch.tensor,
            heading: torch.tensor,
            env_id: int = 0,
            plot_whole_map: bool = False
    ):
        res = self.grid_res  # Grid resolution

        # Create the plot
        plt.figure(figsize=(self.grid_map_padded.shape[0] * res, self.grid_map_padded.shape[1] * res))
        plt.title("Observation semantics visualized")

        # Get points for the environment
        if plot_whole_map:
            points = 1
        points = (idx[env_id] * res).cpu()  # Shape: (N, num_classes)

        # Extract class labels from one-hot encoding
        class_labels = torch.argmax(semantic_map[env_id], dim=1).to(device="cpu")  # Shape: (N,)

        # Define a color for each class
        colors = ["gray", "blue", "green", "yellow", "red", "purple"]
        label = ["Sidewalk", "Crosswalk", "Park", "Street", "Static Obstacles", "Dyn. Obstacles"]

        dot_size = int(1 / res)

        # Plot points for each class
        for class_id in range(len(colors)):
            class_points = points[class_labels == class_id]  # Points belonging to this class
            plt.scatter(class_points[:, 0], class_points[:, 1], c=colors[class_id], s=dot_size, label=f"{label[class_id]}")

        # Robot's postion
        print(f"Robot Position: {robot_pos_idx[env_id]}")
        plt.scatter(robot_pos_idx[env_id][0] * res, robot_pos_idx[env_id][1] * res, c="black", marker="*", s=300, label="Robot Positon")

        # Robot's heading
        length = int(5)
        dx = length * np.cos(heading[env_id])
        dy = length * np.sin(heading[env_id])
        plt.arrow(
            robot_pos_idx[env_id][0] * res, 
            robot_pos_idx[env_id][1] * res,
            dx=dx,
            dy=dy,
            fc="black",
            ec="black",
            head_width=0.5,
            head_length=1, 
            zorder=5
        )

        # Set axis limits and labels
        plt.xlim(0, self.grid_map_padded.shape[0] * res)
        plt.ylim(0, self.grid_map_padded.shape[1] * res)
        plt.xticks(fontsize=6 / res)
        plt.yticks(fontsize=6 / res)
        plt.xlabel("X Coordinate", fontsize=6/res)
        plt.ylabel("Y Coordinate", fontsize=6/res)

        # Add grid and legend
        plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
        plt.legend(fontsize=6/res, loc="upper right")

        # Save the figure as an image
        output_path = f"crowd_navigation_mt/exts/crowd_navigation_mt/crowd_navigation_mt/mdp/observations/output/sem_obs_env_{env_id}_step_{self._env._sim_step_counter}.png"
        plt.savefig(output_path)
        print(f"Map saved to {output_path}")

        plt.close()

    def plot_semantic_map(
        self,
        semantic_map: torch.tensor | None = None, 
    ):
        res = self.grid_res  # Grid resolution

        if semantic_map:
            semantic_map = semantic_map
        else:
            semantic_map = self.grid_map_one_hot

        # Create the plot
        plt.figure(figsize=(self.grid_map_padded.shape[0] * res, self.grid_map_padded.shape[1] * res))
        plt.title("Semantic Map visualized")

        # Extract class labels from one-hot encoding
        class_labels = torch.argmax(semantic_map[env_id], dim=1).to(device="cpu")  # Shape: (N,)

        # Define a color for each class
        colors = ["gray", "blue", "green", "yellow", "red", "purple"]
        label = ["Sidewalk", "Crosswalk", "Park", "Street", "Static Obstacles", "Dyn. Obstacles"]

        dot_size = int(1 / res)

        # Plot points for each class
        for class_id in range(len(colors)):
            class_points = points[class_labels == class_id]  # Points belonging to this class
            plt.scatter(class_points[:, 0], class_points[:, 1], c=colors[class_id], s=dot_size, label=f"{label[class_id]}")
        # Set axis limits and labels
        plt.xlim(0, self.grid_map_padded.shape[0] * res)
        plt.ylim(0, self.grid_map_padded.shape[1] * res)
        plt.xticks(fontsize=6 / res)
        plt.yticks(fontsize=6 / res)
        plt.xlabel("X Coordinate", fontsize=6/res)
        plt.ylabel("Y Coordinate", fontsize=6/res)

        # Add grid and legend
        plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
        plt.legend(fontsize=6/res, loc="upper right")

        # Save the figure as an image
        output_path = f"crowd_navigation_mt/exts/crowd_navigation_mt/crowd_navigation_mt/mdp/observations/output/sem_obs_env_{env_id}_step_{self._env._sim_step_counter}.png"
        plt.savefig(output_path)
        print(f"Map saved to {output_path}")

        plt.close()

    def plot_obs_shape(self, idx: torch.tensor):
        # num_plots = math.ceil(math.sqrt(self.num_envs))         
        num_plots = 2
        fig, axes = plt.subplots(num_plots, num_plots, figsize=(self.grid_map_padded.shape[0]/10,self.grid_map_padded.shape[1]/10))
        fig.suptitle("Observation filter visualized")

        for env_id in range(100,104):            
            # ax = axes[env_id // num_plots, env_id % num_plots]
            ax = axes[(env_id-100) // num_plots, (env_id-100) % num_plots]
            env_points = idx[env_id]
            ax.scatter(env_points[:, 0], env_points[:, 1], c="blue", s=1)
            ax.set_xlim(0, self.grid_map_padded.shape[0])
            ax.set_ylim(0, self.grid_map_padded.shape[1])
            ax.axis("off")
            ax.set_title(f"Env {env_id}", fontsize=6)
            ax.set_xlabel("X Coordinate", fontsize=6)
            ax.set_ylabel("Y Coordinate", fontsize=6)


        # Save the figure as an image
        output_path = "crowd_navigation_mt/exts/crowd_navigation_mt/crowd_navigation_mt/mdp/observations/output/obs_shape_map.png"
        plt.savefig(output_path)
        print(f"Map saved to {output_path}")

        plt.close()


    ##
    # older dyn obstacle implementation$
    ##

    # def _w_to_ego_frame(self, pos_b: torch.tensor, angle: torch.tensor, points_w: torch.tensor):

    #     translated_points = points_w[:, :2] - pos_b.unsqueeze(1)
    #     points_in_ego_frame = translated_points @ self._get_rot_mat(angle)

    #     return points_in_ego_frame

    # def _get_points_in_fov(self, points: torch.tensor):
    #     # TODO: @ vairaviv this gives the points as one dimensional tensor but needed grouped to the environments
    #     points_in_fov = (
    #         (points[:, :, 0] >= -self.obs_x / 2 * self.grid_res) & 
    #         (points[:, :, 0] <= self.obs_x / 2 * self.grid_res) &
    #         (points[:, :, 1] >= -self.obs_y / 2 * self.grid_res) &
    #         (points[:, :, 1] <= self.obs_x / 2 * self.grid_res)
    #     )

    #     return points[points_in_fov]
    
    # def _points_to_map(self, points: torch.tensor):

    #     map_center = (torch.tensor(self.grid_filter_coord.shape, device=self.device) / 2) / self.grid_res

    #     map_idx = torch.div(points, self.grid_res, rounding_mode="trunc").to(device=self.device) + map_center

    #     return map_idx.int()

    # def _add_dynamic_obstacle_to_map(self, pos_b, angle, points_w):
    #     ego_pos = self._w_to_ego_frame(pos_b, angle, points_w)
    #     pos_in_fov = self._get_points_in_fov(ego_pos)
    #     pos_idx = self._points_to_map(pos_in_fov)
    #     return pos_idx
