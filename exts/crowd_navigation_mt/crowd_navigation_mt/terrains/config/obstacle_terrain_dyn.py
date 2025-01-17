# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import crowd_navigation_mt.terrains as terrain_gen_plr
import omni.isaac.lab.terrains as terrain_gen_lab 


from omni.isaac.lab.terrains import TerrainGeneratorCfg

SIZE = 6
OBS_TERRAINS_DYNOBS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=1.0,
    num_rows=SIZE,  # level resolution
    num_cols=SIZE,  # parallel levels
    horizontal_scale=0.125,
    vertical_scale=0.5,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    difficulty_range=(0, 1),
    sub_terrains={
        "obs1": terrain_gen_plr.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0,
            # size=(50, 20),
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.4, 1.6),
            obstacle_height_range=(2, 2),
            platform_width=2.0,
            num_obstacles=5,
        ),
        "obs2": terrain_gen_plr.HfDiscreteObstaclesWedgeTerrainCfg(
            proportion=0.0,
            # size=(50, 20),
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.4, 1.6),
            obstacle_height_range=(2, 2),
            platform_width=2.0,
            num_obstacles=3,
            wedge_depth_range=(2.0, 6.0),
            wedge_thickness=0.5,
            wedge_width_range=(4.0, 5.0),
        ),
    },
)
"""obstacle terrain configuration for dynamic obstacle environments"""
