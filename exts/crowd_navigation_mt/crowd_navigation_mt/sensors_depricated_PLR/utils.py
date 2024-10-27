# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import omni.physics.tensors.impl.api as physx
from omni.isaac.core.prims import XFormPrimView

from omni.isaac.lab.utils.math import convert_quat


def compute_world_poses(
    physxView: XFormPrimView | physx.ArticulationView | physx.RigidBodyView,
    env_ids: torch.Tensor,
    clone: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the world poses of the prim referenced by the prim view.

    Args:
        physxView: The prim view to get the world poses from.
        env_ids: The environment ids of the prims to get the world poses for.

    Raises:
        ValueError: If the prim view is not of the correct type.

    Returns:
        A tuple containing the world positions and orientations of the prims. Orientation is in wxyz format.
    """
    if isinstance(physxView, XFormPrimView):
        pos_w, quat_w = physxView.get_world_poses(env_ids)
    elif isinstance(physxView, physx.ArticulationView):
        pos_w, quat_w = physxView.get_root_transforms()[env_ids].split([3, 4], dim=-1)
        quat_w = convert_quat(quat_w, to="wxyz")
    elif isinstance(physxView, physx.RigidBodyView):
        pos_w, quat_w = physxView.get_transforms()[env_ids].split([3, 4], dim=-1)
        quat_w = convert_quat(quat_w, to="wxyz")
    else:
        raise ValueError(f"Cannot get world poses for prim view of type '{type(physxView)}'.")

    if clone:
        return pos_w.clone(), quat_w.clone()
    else:
        return pos_w, quat_w
