# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from omni.isaac.lab.utils import configclass

from .simple_obst_goal_command import SimpleObstGoalCommand

from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg


"""
Base command generator.
"""
from omni.isaac.lab.envs.mdp.commands import UniformPose2dCommandCfg

from .goal_commands import Uniform2dCoord, RobotGoalCommand, DirectionCommand
from .terrain_analysis import TerrainAnalysis, TerrainAnalysisCfg



@configclass
class SimpleObstGoalCommandCfg(CommandTermCfg):
    """Configuration for the uniform 2D-pose command generator."""

    class_type: type = SimpleObstGoalCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    simple_heading: bool = MISSING
    """Whether to use simple heading or not.

    If True, the heading is in the direction of the target position.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""
        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""
        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the position commands."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_goal"
    )
    """The configuration for the goal pose visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.2, 0.2, 0.8)
    goal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)


###############################################################
# PLR configs
###############################################################

@configclass
class Uniform2dCoordCfg(CommandTermCfg):
    """Configuration for uniform pose command generator.
    This can be used in three ways:
    1. define the full area where all targets are sampled from
        set pos_x and pos_y to the full area

    2. define the sample area around the current position
        set sample_local to True and set the x, y range

    3. define sample are by the environment spacing
        set use_env_spacing to True
    """

    class_type: type = Uniform2dCoord

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    per_env: bool = False
    """If true, the total area is computed by n_envs * x_range * y_range.
    Otherwise, the total area is computed by x_range * y_range."""

    use_env_spacing: bool = False
    """If true, the spacing between the environments is used to determine the range of the area."""

    sample_local: bool = False
    """If true, we sample the next point in a square x y around the current position."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING  # min max [m]
        pos_y: tuple[float, float] = MISSING  # min max [m]

    ranges: Ranges = MISSING

    static: bool = False
    """if true, obstacles do not move"""

    # terrain_analysis: TerrainAnalysisCfg = TerrainAnalysisCfg()
    # """Terrain analysis configuration."""


@configclass
class RobotGoalCommandCfg(CommandTermCfg):
    """Configuration for the robot goal command generator."""

    class_type: type = RobotGoalCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    radius: float = 1
    """Radius of the goal area around the robot."""

    max_goal_distance: float = 25
    """Maximum distance of the goal from the robot."""

    terrain_analysis: TerrainAnalysisCfg | None = None
    """Terrain analysis configuration."""

    angles: list[float] | None = None
    """List of angles in radians to sample from. If None, the angles are sampled uniformly from 0 to 2pi."""

    use_grid_spacing: bool = True
    """If true, the spacing between the terrain cells is used to determine the range of the area.
    Usesfull for grid based terrains with clear spawning area in the middle."""

    deterministic_goal: bool = False
    """If true, the goal is always forward by the same amount."""

    deterministic_goal_distance_x: float = 3.0
    """Distance of the deterministic goal from the robot x."""

    deterministic_goal_distance_y: float = 3.0
    """Distance of the deterministic goal from the robot y."""
    # @configclass
    # class Ranges:
    #     """Uniform distribution ranges for the pose commands."""

    #     radius: tuple[float, float] = MISSING

    robot_to_goal_line_vis: bool = True
    """If true, visualize the line from the robot to the goal."""


@configclass
class DirectionCommandCfg(CommandTermCfg):
    """Configuration for the direction command generator."""

    class_type: type = DirectionCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    direction: tuple[float, float] = MISSING
    """commanded xy direction"""