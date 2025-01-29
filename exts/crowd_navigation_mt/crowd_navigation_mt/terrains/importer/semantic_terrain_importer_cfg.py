# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.terrains import TerrainImporterCfg

from .semantic_terrain_importer import SemanticTerrainImporter

if TYPE_CHECKING:
    from omni.isaac.lab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


@configclass
class SemanticTerrainImporterCfg(TerrainImporterCfg):
    """Configuration for the terrain manager."""

    class_type: type = SemanticTerrainImporter
    """The class to use for the terrain importer.

    Defaults to :class:`crowd_navigation_mt.terrains.importer.SemanticTerrainImporter`.
    """

    semantic_terrain_resolution: float = 0.1

    debug_plot: bool = False
    
