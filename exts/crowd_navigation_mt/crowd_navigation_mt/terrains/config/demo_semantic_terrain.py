# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import crowd_navigation_mt.terrains as terrain_gen_plr
import omni.isaac.lab.terrains as terrain_gen_lab 


from omni.isaac.lab.terrains import TerrainGeneratorCfg

DEMO_SEMANTIC_ENV_CFG = TerrainGeneratorCfg(
    size=(29.0, 20.0),
    border_width=1.0,
    border_height=2.0,
    num_rows=1,  # level resolution
    num_cols=1,  # parallel levels
    horizontal_scale=0.1,
    vertical_scale=0.1,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(0, 1),
    # curriculum_shuffle=True,
    sub_terrains={
        "obs1": terrain_gen_lab.HfDiscreteObstaclesTerrainCfg(
            proportion=0.3,  # (N_PARALLEL_TERRAINS - 2) / N_PARALLEL_TERRAINS,
            # size=(50, 20),
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.4, 1.6),
            obstacle_height_range=(0.4, 2),
            platform_width=2.0,
            num_obstacles=8,  # 8
        ),
    }
)