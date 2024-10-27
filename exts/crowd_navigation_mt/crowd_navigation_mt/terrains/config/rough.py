# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import crowd_navigation_mt.terrains as terrain_gen_plr
import omni.isaac.lab.terrains as terrain_gen_lab


from omni.isaac.lab.terrains import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen_lab.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen_lab.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen_lab.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen_lab.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen_lab.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen_lab.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""

N_PARALLEL_TERRAINS = 24  # 30
OBS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=24,  # level resolution
    num_cols=N_PARALLEL_TERRAINS,  # parallel levels
    horizontal_scale=0.125,
    vertical_scale=0.5,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    difficulty_range=(0, 1),
    # curriculum_shuffle=True,
    sub_terrains={
        "obs1": terrain_gen_lab.HfDiscreteObstaclesTerrainCfg(
            proportion=0.3,  # (N_PARALLEL_TERRAINS - 2) / N_PARALLEL_TERRAINS,
            # size=(50, 20),
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.4, 1.6),
            obstacle_height_range=(2, 2),
            platform_width=2.0,
            num_obstacles=8,  # 8
        ),
        "obs2": terrain_gen_plr.HfDiscreteObstaclesWedgeTerrainCfg(
            proportion=0.05,
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
        "obs3": terrain_gen_plr.HfDiscreteObstaclesCellSideTerrainCfg(
            proportion=0.35,
            # size=(50, 20),
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.4, 1.6),
            obstacle_height_range=(2, 2),
            platform_width=2.0,
            num_obstacles=6,
            cell_wall_width=2.0,
            cell_wall_thickness=0.5,
            position_range=(-3, 3),
        ),
        "obs4": terrain_gen_plr.HfDiscreteLShapedPassageTerrainCfg(
            proportion=0.05,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.4, 1.6),
            obstacle_height_range=(2, 2),
            platform_width=1.5,
            num_obstacles=6,
            l_width_range=(2, 2.5),
            l_side_range=(3.5, 4.75),
        ),
        "obs5": terrain_gen_plr.HfDiscreteSShapedPassageTerrainCfg(
            proportion=0.25,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.4, 1.6),
            obstacle_height_range=(2, 2),
            platform_width=1.25,
            num_obstacles=6,
            s_width_range=(2.5, 2.75),
            s_side_range=(2.5, 3),
            s_length_range=(5, 6),
            closes_prob=0.8,
        ),
    },
)

SIZE = 6
OBS_TERRAINS_DYNOBS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=SIZE,  # level resolution
    num_cols=SIZE,  # parallel levels
    horizontal_scale=0.125,
    vertical_scale=0.5,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
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
"""obstacle terrain configuration."""


SIZE = 8
TEST_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=SIZE,  # level resolution
    num_cols=SIZE,  # parallel levels
    horizontal_scale=0.125,
    vertical_scale=0.5,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(1, 1),
    sub_terrains={
        "obs1": terrain_gen_plr.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0,
            # size=(50, 20),
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.5, 1.6),
            obstacle_height_range=(2, 2),
            platform_width=2.0,
            num_obstacles=12,
        ),
        "obs5": terrain_gen_plr.HfDiscreteSShapedPassageTerrainCfg(
            proportion=0.0,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.4, 1.6),
            obstacle_height_range=(2, 2),
            platform_width=1.5,
            num_obstacles=6,
            s_width_range=(2.5, 2.75),
            s_side_range=(2.5, 3),
            s_length_range=(5, 6),
            closes_prob=0.8,
        ),
    },
)
"""test terrain configuration."""
