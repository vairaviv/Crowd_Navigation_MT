# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import trimesh
from typing import TYPE_CHECKING

import warp
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.utils.warp import convert_to_warp_mesh

from omni.isaac.lab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg
from omni.isaac.lab.terrains.terrain_importer import TerrainImporter, TerrainGenerator
from omni.isaac.lab.terrains.trimesh.utils import make_plane
from omni.isaac.lab.terrains.utils import create_prim_from_mesh




if TYPE_CHECKING:
    from .semantic_terrain_importer_cfg import SemanticTerrainImporterCfg


class SemanticTerrainImporter(TerrainImporter):
    r"""A class adding semantic map of the terrain
    curretnly only for demo purposes
    """

    grid_map: torch.tensor
    """Grid with semantic informations"""
    grid_map_one_hot: torch.tensor

    cfg: SemanticTerrainImporterCfg

    def __init__(self, cfg: SemanticTerrainImporterCfg):
        super().__init__(cfg=cfg)
        
        from ..elevation_map.semantic_height_map import create_semantic_map

        self.grid_map = create_semantic_map(self.device, self.cfg.terrain_generator.size, self.cfg)
        num_classes = torch.unique(self.grid_map).size(0)
        self.grid_map_one_hot = torch.nn.functional.one_hot(self.grid_map, num_classes)

        # transform vector in order to shift the gridmap to world frame, scaled for transform
        width = cfg.terrain_generator.size[0]
        height = cfg.terrain_generator.size[1]
        self.transform_vector = torch.tensor([-width / 2, -height / 2]).to(device=self.device)


