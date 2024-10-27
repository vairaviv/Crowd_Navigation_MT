# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass


@dataclass
class RayCasterData:
    """Data container for the ray-cast sensor."""

    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame.

    Shape is (N, 3), where N is the number of sensors.
    """
    quat_w: torch.Tensor = None
    """Orientation of the sensor origin in quaternion (w, x, y, z) in world frame.

    Shape is (N, 4), where N is the number of sensors.
    """
    vel_w: torch.Tensor = None
    """Velocity of the sensor origin in world frame.
    
    Shape is (N, 3), where N is the number of sensors.
    """

    rot_vel_w: torch.Tensor = None
    """Angular velocity of the sensor origin in world frame.
    
    Shape is (N, 3), where N is the number of sensors.
    """

    ray_hits_w: torch.Tensor = None
    """The ray hit positions in the world frame.

    Shape is (N, B, 3), where N is the number of sensors, B is the number of rays
    in the scan pattern per sensor.
    """

    ray_hit_mesh_idx: torch.Tensor = None
    """The mesh index of the ray hit positions.
    
    Shape is (N, B), where N is the number of sensors, B is the number of rays
    entries are integers representing the mesh index of the ray hit positions.
    -1 indicates no hit.
    """

    distances: torch.Tensor = None
    """The distances of the ray hit positions from the sensor origin."""

    mesh_positions_w: torch.Tensor | None = None
    """The mesh positions in the world frame. Shape is (M, N, 3), where M is the number of meshes, N is the number of sensors.

    Note, this will not be updated if static_meshes is set to True in the ray caster config.
    """
    mesh_orientations_w: torch.Tensor | None = None
    """The mesh orientations in the world frame. Shape is (M, N, 4), where M is the number of meshes, N is the number of sensors.

    Note, this will not be updated if static_meshes is set to True in the ray caster config.
    """

    mesh_velocities_w: torch.Tensor | None = None
    """The mesh velocities in the world frame. Shape is (M, N, 3), where M is the number of meshes, N is the number of sensors.
    
    Note, this will not be updated if static_meshes is set to True in the ray caster config.
    """

    mesh_angular_velocities_w: torch.Tensor | None = None
    """The mesh angular velocities in the world frame. Shape is (M, N, 3), where M is the number of meshes, N is the number of sensors.
    
    Note, this will not be updated if static_meshes is set to True in the ray caster config.
    """
