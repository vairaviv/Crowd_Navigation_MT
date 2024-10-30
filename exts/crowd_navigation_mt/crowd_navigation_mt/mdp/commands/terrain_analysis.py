# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import os
import pickle
import scipy.spatial.transform as tf
import torch
from dataclasses import MISSING
from scipy.spatial import KDTree
from scipy.stats import qmc, mode

import networkx as nx
import warp as wp
#from skimage.draw import line

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.sensors import RayCaster, patterns, RayCasterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.warp import raycast_dynamic_meshes


@configclass
class TerrainAnalysisCfg:
    robot_height: float = 0.8
    """Height of the robot"""
    wall_height: float = 3.0
    """Height of the walls."""
    min_wall_dist: float = 1
    """minimum distance to consider a wall"""
    sample_dist: float = 4
    """dist to sample around"""
    raycaster_sensor: str = MISSING
    """Name of the raycaster sensor to use for terrain analysis"""


class TerrainAnalysis:
    def __init__(self, cfg: TerrainAnalysisCfg, env: ManagerBasedRLEnv):
        # save cfg and env
        self.cfg = cfg
        self._env = env

        # get the raycaster sensor that should be used to raycast against all the ground meshes
        if isinstance(self._env.scene.sensors[self.cfg.raycaster_sensor], RayCaster):
            self._raycaster: RayCaster = self._env.scene.sensors[self.cfg.raycaster_sensor]
        else:
            raise ValueError(f"Sensor {self.cfg.raycaster_sensor} is not a RayCaster sensor")

        # TODO make this work
        scan_2d_pattern = patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(0, 360),
            horizontal_res=20,
        )
        self.scan_2d_directions = patterns.lidar_pattern(scan_2d_pattern, env.device)[1]

        height_scan_pattern = patterns.GridPatternCfg(resolution=0.333, size=[1.5, 1.5])
        self.height_scan_starts, self.height_scan_directions = patterns.grid_pattern(height_scan_pattern, env.device)

    def sample_spawn(self, init_spawn_point2d: torch.Tensor) -> torch.Tensor:
        height = self.sample_spawn_height(init_spawn_point2d)
        point_3d = torch.cat((init_spawn_point2d, height.unsqueeze(1)), dim=1)

        return self.sample_spawn_2d(point_3d)

    def sample_spawn_height(self, spawn_2d: torch.Tensor, use_mode: bool = True, use_max: bool = False) -> torch.Tensor:

        z_values = torch.ones(len(spawn_2d), device=self._env.device).unsqueeze(1) * 50
        init_points = torch.cat((spawn_2d, z_values), dim=1).unsqueeze(1)

        ray_starts = self.height_scan_starts.unsqueeze(0).repeat(len(spawn_2d), 1, 1) + init_points.repeat(
            1, len(self.height_scan_starts), 1
        )
        
        # TODO add mesh_ids_wp
        z_positions = raycast_dynamic_meshes(
            ray_starts=ray_starts.to(torch.float32),
            ray_directions=self.height_scan_directions.unsqueeze(0).repeat(len(spawn_2d), 1, 1).to(torch.float32),
            mesh_ids_wp=self._raycaster._mesh_ids_wp,
            #meshes=np.tile(np.array(self._raycaster._meshes[0], dtype=wp.Mesh)[0], (len(spawn_2d), 1)),
            max_dist=1000,
            return_distance=False,
        )[0][..., 2]

        if use_mode:
            return (
                torch.tensor(mode(z_positions.cpu(), keepdims=False, axis=1).mode, device=self._env.device)
                + self.cfg.robot_height
            )
        elif use_max:
            return torch.max(z_positions, dim=1).values + self.cfg.robot_height

        return torch.mean(z_positions, dim=1) + self.cfg.robot_height

    def sample_spawn_2d(self, init_spawn_point: torch.Tensor, num_resamples: int = 100) -> torch.Tensor:
        """given a spawn point, sample a valid spawn point around it."""

        halton_sampler = qmc.Halton(d=2, scramble=True)
        sample_points = halton_sampler.random(num_resamples)
        sample_points = (sample_points - 0.5) * self.cfg.sample_dist * 2
        sample_points = torch.from_numpy(sample_points).to(self._env.device)
        sample_points = sample_points.unsqueeze(0).expand(len(init_spawn_point), num_resamples, 2)

        init_points = init_spawn_point[:, :2].unsqueeze(1)
        sample_points_w = sample_points + init_points

        z_dim = init_spawn_point[:, 2].unsqueeze(1).expand(len(init_spawn_point), num_resamples).unsqueeze(2)

        sample_points_3d = torch.cat((sample_points_w, z_dim), dim=2)

        samples = torch.zeros(len(init_spawn_point), 3, device=self._env.device)

        for env_id, env_samples in enumerate(sample_points_3d):
            # per environment, find all valid points and select one randomly
            en_valids = self._point_valid(env_samples)
            valid_poin_idx = torch.where(en_valids)[0]
            if len(valid_poin_idx) == 0:
                raise ValueError("No valid spawn points found")

            sample_id = valid_poin_idx[torch.randint(0, len(valid_poin_idx), (1,))]
            samples[env_id] = env_samples[sample_id]

        return samples

    def _point_valid(self, sample_point: torch.Tensor) -> bool:
        # raycast with the 2d lidar scan. we scan all n_samples at once (like multiple envs)
        # shapes of inputs are (n_samples, n_rays, 3)
        # TODO check if this even makes sense
        N = 16
        if self._env.num_envs <= N:
            N = self._env.num_envs
            
        closest_meshes_list, keep_indices = self._extract_n_closest_meshes(sample_point.mean(dim=0), N=N)
        mesh_positions = (
            self._raycaster._data.mesh_positions_w[0][keep_indices].unsqueeze(0).repeat(len(sample_point), 1, 1).to(torch.float32)
        )
        mesh_orientations = (
            self._raycaster._data.mesh_orientations_w[0][keep_indices].unsqueeze(0).repeat(len(sample_point), 1, 1).to(torch.float32)
        )

        # TODO add mesh_ids_wp
        scan_2d_dists = raycast_dynamic_meshes(
            ray_starts=sample_point.unsqueeze(1).repeat(1, len(self.scan_2d_directions), 1).to(torch.float32),
            ray_directions=self.scan_2d_directions.unsqueeze(0).repeat(len(sample_point), 1, 1).to(torch.float32),
            # meshes=np.tile(np.array(self._raycaster._meshes[0], dtype=wp.Mesh)[0], (len(sample_point), 1)),
            mesh_ids_wp=self._raycaster._mesh_ids_wp,
            # meshes=np.tile(np.array(closest_meshes_list, dtype=wp.Mesh), (len(sample_point), 1)),
            mesh_positions_w=mesh_positions,
            mesh_orientations_w=mesh_orientations,
            max_dist=5,
            return_distance=True,
        )[1]
        # point is valid if all distances are greater than min_wall_dist
        return (scan_2d_dists > self.cfg.min_wall_dist).all(dim=1)

    def _extract_n_closest_meshes(self, center_point: torch.tensor, N: int, mesh_ids_to_keep: list[int] = [0]):
        # Adjust N based on the length of mesh_ids_to_keep
        N -= len(mesh_ids_to_keep)

        # Convert mesh_ids_to_keep to a tensor and repeat it for each robot
        mesh_ids_to_keep_: torch.Tensor = torch.tensor(mesh_ids_to_keep, device=center_point.device)

        # Extract mesh positions and robot positions based on env_ids
        mesh_pos = self._raycaster._data.mesh_positions_w[0]  # w positions are the same in all envs
        meshes = self._raycaster._meshes[0]  # meshes are the same in all envs

        # Compute squared distances
        distances = torch.sum((mesh_pos - center_point) ** 2, dim=1)

        # Get the indices of the N closest meshes for each robot
        _, closest_indices = torch.topk(distances, N, largest=False, sorted=True)

        # Combine mesh_ids_to_keep with closest_indices
        keep_indices = torch.unique(torch.cat((mesh_ids_to_keep_, closest_indices), dim=0))

        # Construct the list of lists of closest meshes
        closest_meshes_list = [meshes[i] for i in keep_indices]

        return closest_meshes_list, keep_indices
