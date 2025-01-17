# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import crowd_navigation_mt.terrains as terrain_gen_plr
import omni.isaac.lab.terrains as terrain_gen_lab 


from omni.isaac.lab.terrains import TerrainGeneratorCfg

N_PARALLEL_TERRAINS = 6  # 30
N_STAT_LEVEL = 1
N_DYN_LEVEL = 1
CORRIDOR_WITH_STAT_OBS = TerrainGeneratorCfg(
    size=(4.0, 20.0),
    border_width=1.0,
    border_height=2.0,
    num_rows=N_STAT_LEVEL,  # level resolution
    num_cols=N_DYN_LEVEL,  # parallel levels
    horizontal_scale=0.125,
    vertical_scale=0.5,
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
            obstacle_width_range=(0.4, 0.5),
            obstacle_height_range=(0.4, 2),
            platform_width=2.0,
            num_obstacles=8,  # 8
        ),
        # "obs2": terrain_gen_plr.HfDiscreteObstaclesWedgeTerrainCfg(
        #     proportion=0.05,
        #     # size=(50, 20),
        #     obstacle_height_mode="fixed",
        #     obstacle_width_range=(0.4, 1.6),
        #     obstacle_height_range=(2, 2),
        #     platform_width=2.0,
        #     num_obstacles=3,
        #     wedge_depth_range=(2.0, 6.0),
        #     wedge_thickness=0.5,
        #     wedge_width_range=(4.0, 5.0),
        # ),
        # "obs3": terrain_gen_plr.HfDiscreteObstaclesCellSideTerrainCfg(
        #     proportion=0.35,
        #     # size=(50, 20),
        #     obstacle_height_mode="fixed",
        #     obstacle_width_range=(0.4, 1.6),
        #     obstacle_height_range=(2, 2),
        #     platform_width=2.0,
        #     num_obstacles=6,
        #     cell_wall_width=2.0,
        #     cell_wall_thickness=0.5,
        #     position_range=(-3, 3),
        # ),
        # "obs4": terrain_gen_plr.HfDiscreteLShapedPassageTerrainCfg(
        #     proportion=0.05,
        #     obstacle_height_mode="fixed",
        #     obstacle_width_range=(0.4, 1.6),
        #     obstacle_height_range=(2, 2),
        #     platform_width=1.5,
        #     num_obstacles=6,
        #     l_width_range=(2, 2.5),
        #     l_side_range=(3.5, 4.75),
        # ),
        # "obs5": terrain_gen_plr.HfDiscreteSShapedPassageTerrainCfg(
        #     proportion=0.25,
        #     obstacle_height_mode="fixed",
        #     obstacle_width_range=(0.4, 1.6),
        #     obstacle_height_range=(2, 2),
        #     platform_width=1.25,
        #     num_obstacles=6,
        #     s_width_range=(2.5, 2.75),
        #     s_side_range=(2.5, 3),
        #     s_length_range=(5, 6),
        #     closes_prob=0.8,
        # ),
    },
)