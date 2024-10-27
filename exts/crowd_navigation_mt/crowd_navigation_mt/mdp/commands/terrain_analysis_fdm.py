# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import os
import pickle
import random
import scipy.spatial.transform as tf
import torch
from dataclasses import MISSING
from scipy.spatial import KDTree
from scipy.stats import qmc

import networkx as nx
import warp as wp
from skimage.draw import line

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.sensors import RayCaster
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.warp import raycast_dynamic_meshes


@configclass
class TerrainAnalysisCfg:
    robot_height: float = 0.6
    """Height of the robot"""
    wall_height: float = 3.0
    """Height of the walls.

    Wall filtering will start rays from that height and filter all that hit the mesh within 0.3m."""
    robot_buffer_spawn: float = 0.7
    """Robot buffer for spawn location"""
    tree_nodes: int = 15000
    """Number of nodes in the tree"""
    max_path_length: float = 10.0
    """Maximum distance from the start location to the goal location"""
    num_connections: int = 5
    """Number of connections to make in the graph"""
    raycaster_sensor: str = MISSING
    """Name of the raycaster sensor to use for terrain analysis"""
    grid_resolution: float = 0.1
    """Resolution of the grid to check for not traversable edges"""
    height_diff_threshold: float = 0.3
    """Threshold for height difference between two points"""
    viz_graph: bool = True
    """Visualize the graph after the construction for a short amount of time."""


class TerrainAnalysis:
    def __init__(self, cfg: TerrainAnalysisCfg, env: ManagerBasedRLEnv):
        # save cfg and env
        self.cfg = cfg
        self._env = env

    def sample_paths(self, num_paths, min_path_length, max_path_length, seed: int = 1) -> torch.Tensor:
        # check dimensions
        assert (
            len(num_paths) == len(min_path_length) == len(max_path_length)
        ), "Number of paths, min path length and max path length must be equal"

        # the data is stored in torch tensors with the structure
        # [start_x, start_y, start_z, goal_x, goal_y, goal_z, path_length]
        data = torch.empty(0, 7)

        # load paths if they exist
        num_paths_to_explore = []
        min_path_length_to_explore = []
        max_path_length_to_explore = []
        for num_path, min_len, max_len in zip(num_paths, min_path_length, max_path_length):
            filename = self._get_save_path(seed, num_path, min_len, max_len)
            if os.path.isfile(filename):
                with open(filename, "rb") as f:
                    saved_paths = pickle.load(f)
                # add loaded path dict to data dict
                data = torch.concatenate((data, saved_paths))
                print(f"[INFO] Loaded {num_path} with [{min_len},{max_len}] length generated with seed {seed}.")
            else:
                num_paths_to_explore.append(num_path)
                min_path_length_to_explore.append(min_len)
                max_path_length_to_explore.append(max_len)

        if len(num_paths_to_explore) == 0:
            return data

        # construct graph if samples not loaded
        if not hasattr(self, "graph"):
            self._sample_points()
            self._construct_graph()

        # map distance to idx pairs
        random.seed(seed)

        for num_path, min_len, max_len in zip(
            num_paths_to_explore, min_path_length_to_explore, max_path_length_to_explore
        ):
            # get index of samples within length
            within_length = (self.samples[:, 2] > min_len) & (self.samples[:, 2] <= max_len)

            # randomly select certain pairs
            rand_idx = torch.randperm(self.samples.shape[0])

            # select the samples
            selected_samples = self.samples[rand_idx[within_length]][:num_path]

            # get start, goal and path length
            curr_data = torch.zeros((num_path, 7))
            curr_data[:, :3] = self.points[selected_samples[:, 0].type(torch.int64)]
            curr_data[:, 3:6] = self.points[selected_samples[:, 1].type(torch.int64)]
            curr_data[:, 6] = selected_samples[:, 2]

            # save curr_data as pickle
            filename = self._get_save_path(seed, num_path, min_len, max_len)
            with open(filename, "wb") as f:
                pickle.dump(curr_data, f)

            # update data buffer
            data = torch.concatenate((data, curr_data), dim=0)

        # define start points
        return data

    ###
    # Helper functions
    ###

    def _sample_points(self):
        # get mesh dimensions
        x_max, y_max, x_min, y_min = self._get_mesh_dimensions()

        # init sampler as qmc
        sampler = qmc.Halton(d=2, scramble=False)
        sampled_nb_points = 0
        sampled_points = []

        # get the raycaster sensor that should be used to raycast against all the ground meshes
        if isinstance(self._env.scene.sensors[self.cfg.raycaster_sensor], RayCaster):
            self._raycaster: RayCaster = self._env.scene.sensors[self.cfg.raycaster_sensor]
        else:
            raise ValueError(f"Sensor {self.cfg.raycaster_sensor} is not a RayCaster sensor")

        while sampled_nb_points < self.cfg.tree_nodes:
            # get raw samples origins
            points = sampler.random(self.cfg.tree_nodes)
            points = qmc.scale(points, [x_min, y_min], [x_max, y_max])
            heights = np.ones((self.cfg.tree_nodes, 1)) * self.cfg.wall_height

            ray_origins = torch.from_numpy(np.hstack((points, heights))).type(torch.float32)

            # filter points that are outside the mesh or inside walls
            ray_origins, z_depth, heights = self._point_filter_wall(ray_origins, torch.tensor(heights))

            # filter points that are too close to walls
            ray_origins = self._point_filter_wall_closeness(ray_origins, heights, z_depth)

            sampled_points.append(torch.clone(ray_origins))
            sampled_nb_points += ray_origins.shape[0]

        self.points = torch.vstack(sampled_points)
        self.points = self.points[: self.cfg.tree_nodes]
        return

    def _construct_graph(self):
        # construct kdtree to find nearest neighbors of points
        kdtree = KDTree(self.points.cpu().numpy())
        _, nearest_neighbors_idx = kdtree.query(self.points.cpu().numpy(), k=self.cfg.num_connections + 1, workers=-1)
        # remove first neighbor as it is the point itself
        nearest_neighbors_idx = torch.tensor(nearest_neighbors_idx[:, 1:], dtype=torch.int64)

        # filter connections that collide with the environment
        idx_edge_start, idx_edge_end, distance = self._edge_filter_mesh_collisions(nearest_neighbors_idx)

        idx_edge_start, idx_edge_end, distance, idx_edge_start_filtered, idx_edge_end_filtered = (
            self._edge_filter_height_diff(idx_edge_start, idx_edge_end, distance)
        )

        # init graph
        self.graph = nx.Graph()
        # add nodes with position attributes
        self.graph.add_nodes_from(list(range(self.cfg.tree_nodes)))
        pos_attr = {i: {"pos": self.points[i].cpu().numpy()} for i in range(self.cfg.tree_nodes)}
        nx.set_node_attributes(self.graph, pos_attr)
        # add edges with distance attributes
        # NOTE: as the shortest path searching algorithm only stores integers
        self.graph.add_edges_from(list(map(tuple, np.stack((idx_edge_start, idx_edge_end), axis=1))))
        distance_attr = {
            (i, j): {"distance": distance[idx]} for idx, (i, j) in enumerate(zip(idx_edge_start, idx_edge_end))
        }
        nx.set_edge_attributes(self.graph, distance_attr)

        # get all shortest paths
        odom_goal_distances = dict(
            nx.all_pairs_dijkstra_path_length(self.graph, cutoff=self.cfg.max_path_length, weight="distance")
        )

        # summarize to samples
        samples = []
        for key, value in odom_goal_distances.items():
            curr_samples = torch.zeros((len(value), 3))
            curr_samples[:, 0] = key
            curr_samples[:, 1] = torch.tensor(list(value.keys()))
            curr_samples[:, 2] = torch.tensor(list(value.values()))
            samples.append(curr_samples)
        self.samples = torch.vstack(samples)

        # debug visualization
        if self.cfg.viz_graph:
            # in headless mode, we cannot visualize the graph and omni.debug.draw is not available
            try:
                import omni.isaac.debug_draw._debug_draw as omni_debug_draw

                draw_interface = omni_debug_draw.acquire_debug_draw_interface()
                draw_interface.draw_points(
                    self.points.tolist(),
                    [(1.0, 0.5, 0, 1)] * self.cfg.tree_nodes,
                    [5] * self.cfg.tree_nodes,
                )
                for start_idx, goal_idx in zip(idx_edge_start, idx_edge_end):
                    draw_interface.draw_lines(
                        [self.points[start_idx].tolist()],
                        [self.points[goal_idx].tolist()],
                        [(0, 1, 0, 1)],
                        [1],
                    )
                for start_idx, goal_idx in zip(idx_edge_start_filtered, idx_edge_end_filtered):
                    draw_interface.draw_lines(
                        [self.points[start_idx].tolist()],
                        [self.points[goal_idx].tolist()],
                        [(1, 0, 0, 1)],
                        [1],
                    )
                for _ in range(3000):
                    self._env.sim.render()

                # clear the drawn points and lines
                draw_interface.clear_points()
                draw_interface.clear_lines()

            except ImportError:
                print("[WARNING] Graph Visualization is not available in headless mode.")

    def _get_mesh_dimensions(self) -> tuple[float, float, float, float]:
        # get min, max of the mesh in the xy plane
        # Get bounds of the terrain
        bounds = []
        for mesh in self._env.scene.terrain.meshes.values():
            bounds.append(mesh.bounds)
        bounds = np.vstack(bounds)
        x_min, y_min = np.min(bounds[:, 0]), np.min(bounds[:, 1])
        x_max, y_max = np.max(bounds[:, 0]), np.max(bounds[:, 1])
        return x_max, y_max, x_min, y_min

    def _get_save_path(self, seed, num_path: int, min_len: float, max_len: float) -> str:
        filename = f"paths_seed{seed}_paths{num_path}_min{min_len}_max{max_len}.pkl"
        # get env name
        assert isinstance(self._env.scene.terrain.cfg.usd_path, str), "Only works with environments loaded from usd!"
        env_name = os.path.splitext(self._env.scene.terrain.cfg.usd_path)[0]
        # create directory if necessary
        filedir = os.path.join(self._env.scene.terrain.cfg.usd_path, env_name)
        os.makedirs(filedir, exist_ok=True)
        return os.path.join(filedir, filename)

    ###
    # Point filter functions
    ###

    def _point_filter_wall(
        self, ray_origins: torch.Tensor, heights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # get ray directions in negative z direction
        ray_directions = torch.zeros((self.cfg.tree_nodes, 3), dtype=torch.float32)
        ray_directions[:, 2] = -1.0

        # TODO add mesh_ids_wp
        z_depth = raycast_dynamic_meshes(
            ray_starts=ray_origins.unsqueeze(0),
            ray_directions=ray_directions.unsqueeze(0),
            meshes=np.array(self._raycaster._meshes, dtype=wp.Mesh),
            max_dist=5,
            return_distance=True,
        )[1].squeeze(0)

        # filter points outside the mesh and within walls
        filter_inside_mesh = torch.isfinite(z_depth)  # outside mesh
        print(f"[INFO] filtered {self.cfg.tree_nodes - filter_inside_mesh.sum()} points outside of mesh")
        filter_outside_wall = z_depth > 0.3  # inside wall
        print(f"[INFO] filtered {self.cfg.tree_nodes - filter_outside_wall.sum()} points inside wall")
        filter_combined = torch.all(torch.stack((filter_inside_mesh, filter_outside_wall), dim=1), dim=1)
        print(
            f"[INFO] filtered total of {round(float((1 - filter_combined.sum() / self.cfg.tree_nodes) * 100), 4)}"
            " % of points"
        )

        return ray_origins[filter_combined].type(torch.float32), z_depth[filter_combined], heights[filter_combined]

    def _point_filter_wall_closeness(
        self, ray_origins: torch.Tensor, heights: torch.Tensor, z_depth: torch.Tensor
    ) -> torch.Tensor:
        # reduce ground height to check for closeness to walls and other objects
        ray_origins[:, 2] = heights[:, 0] - z_depth + self.cfg.robot_height
        # enforce a minimum distance to the walls
        angles = np.linspace(-np.pi, np.pi, 20)
        ray_directions = tf.Rotation.from_euler("z", angles, degrees=False).as_matrix() @ np.array([1, 0, 0])
        ray_hit = []

        for ray_direction in ray_directions:
            ray_direction_torch = torch.from_numpy(ray_direction).repeat(ray_origins.shape[0], 1).type(torch.float32)
            # TODO add mesh_ids_wp
            distance = raycast_dynamic_meshes(
                ray_starts=ray_origins.unsqueeze(0),
                ray_directions=ray_direction_torch.unsqueeze(0),
                meshes=np.array(self._raycaster._meshes, dtype=wp.Mesh),
                max_dist=self.cfg.robot_buffer_spawn,
                return_distance=True,
            )[1].squeeze(0)
            ray_hit.append(torch.isinf(distance))

        # check if every point has the minimum distance in every direction
        without_wall = torch.all(torch.vstack(ray_hit), dim=0)

        print(f"[INFO] filtered {ray_origins.shape[0] - without_wall.sum().item()} points too close to walls")
        ray_origins = ray_origins[without_wall].type(torch.float32)

        return ray_origins

    ###
    # Edge filtering functions
    ###

    def _edge_filter_height_diff(
        self, idx_edge_start: np.ndarray, idx_edge_end: np.ndarray, distance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Filter edges based on height difference between points."""
        # get dimensions and construct height grid with raycasting
        x_max, y_max, x_min, y_min = self._get_mesh_dimensions()
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(x_min, x_max, int((x_max - x_min) / self.cfg.grid_resolution)),
            torch.linspace(y_min, y_max, int((y_max - y_min) / self.cfg.grid_resolution)),
        )
        grid_z = torch.ones_like(grid_x) * 10
        grid_points = torch.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T
        direction = torch.zeros_like(grid_points)
        direction[:, 2] = -1.0

        # check for collision with raycasting
        # TODO add mesh_ids_wp
        hit_point = raycast_dynamic_meshes(
            ray_starts=grid_points.unsqueeze(0),
            ray_directions=direction.unsqueeze(0),
            meshes=np.array(self._raycaster._meshes, dtype=wp.Mesh),
            max_dist=15,
        )[0].squeeze(0)

        height_grid = hit_point[:, 2].reshape(
            int((x_max - x_min) / self.cfg.grid_resolution), int((y_max - y_min) / self.cfg.grid_resolution)
        )

        # compute height difference
        height_diff = torch.diff(height_grid, dim=0, append=torch.zeros(1, height_grid.shape[1])) + torch.diff(
            height_grid, dim=1, append=torch.zeros(height_grid.shape[0], 1)
        )
        height_diff = np.abs(height_diff.cpu().numpy()) > self.cfg.height_diff_threshold

        # identify which edges are on different heights
        edge_idx = torch.abs(self.points[idx_edge_start, 2] - self.points[idx_edge_end, 2]) > 0.1

        # filter edges that are on different heights
        check_idx_edge_start = idx_edge_start[edge_idx]
        check_idx_edge_end = idx_edge_end[edge_idx]

        check_grid_idx_start = (
            ((self.points[check_idx_edge_start, :2] - torch.tensor([x_min, y_min])) / self.cfg.grid_resolution)
            .int()
            .cpu()
            .numpy()
        )
        check_grid_idx_end = (
            ((self.points[check_idx_edge_end, :2] - torch.tensor([x_min, y_min])) / self.cfg.grid_resolution)
            .int()
            .cpu()
            .numpy()
        )

        filter_idx = np.zeros(check_idx_edge_start.shape[0], dtype=bool)

        for idx, (edge_start_idx, edge_end_idx) in enumerate(zip(check_grid_idx_start, check_grid_idx_end)):
            grid_idx_x, grid_idx_y = line(edge_start_idx[0], edge_start_idx[1], edge_end_idx[0], edge_end_idx[1])

            filter_idx[idx] = np.any(height_diff[grid_idx_x, grid_idx_y])

        # set the indexes that should be removed in edge_idx to true
        edge_idx[edge_idx.clone()] = torch.tensor(filter_idx)
        edge_idx = edge_idx.cpu().numpy()
        # filter edges
        idx_edge_start_filtered = idx_edge_start[edge_idx]
        idx_edge_end_filtered = idx_edge_end[edge_idx]

        idx_edge_start = idx_edge_start[~edge_idx]
        idx_edge_end = idx_edge_end[~edge_idx]
        distance = distance[~edge_idx]

        return idx_edge_start, idx_edge_end, distance, idx_edge_start_filtered, idx_edge_end_filtered

    def _edge_filter_mesh_collisions(
        self, nearest_neighbors_idx: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter connections that collide with the environment."""
        # define origin and neighbor points
        origin_point = torch.repeat_interleave(self.points, repeats=self.cfg.num_connections, axis=0)
        # TODO replace: origin_point = torch.repeat_interleave(self.points, repeats=self.cfg.num_connections)
        neighbor_points = self.points[nearest_neighbors_idx, :].reshape(-1, 3)
        min_distance = torch.norm(origin_point - neighbor_points, dim=1)

        # check for collision with raycasting
        # TODO add mesh_ids_wp
        distance = raycast_dynamic_meshes(
            ray_starts=origin_point.unsqueeze(0),
            ray_directions=(origin_point - neighbor_points).unsqueeze(0),
            meshes=np.array(self._raycaster._meshes, dtype=wp.Mesh),
            max_dist=self.cfg.max_path_length,
            return_distance=True,
        )[1]

        distance[torch.isinf(distance)] = self.cfg.max_path_length
        # filter connections that collide with the environment
        collision = (distance < min_distance).reshape(-1, self.cfg.num_connections)

        # get edge indices
        idx_edge_start = np.repeat(np.arange(self.cfg.tree_nodes), repeats=self.cfg.num_connections, axis=0)
        idx_edge_end = nearest_neighbors_idx.reshape(-1).cpu().numpy()

        # filter collision edges and distances
        idx_edge_end = idx_edge_end[~collision.reshape(-1).cpu().numpy()]
        idx_edge_start = idx_edge_start[~collision.reshape(-1).cpu().numpy()]
        distance = min_distance[~collision.reshape(-1)].cpu().numpy()

        return idx_edge_start, idx_edge_end, distance
