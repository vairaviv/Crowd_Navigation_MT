

from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.managers import ActionTerm, ActionManager, ActionTermCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from omni.isaac.lab.utils import configclass


from dataclasses import MISSING

import torch
import math
import crowd_navigation_mt.mdp as mdp
from .obstacle_actions import SimpleDynObstacleActionTerm


@configclass
class SimpleDynObstacleActionTermCfg(ActionTermCfg):
    """Configuration for the simple obstacle action term."""

    class_type: type = SimpleDynObstacleActionTerm
    """The class corresponding to the action term."""

    max_velocity: float = 5.0

    max_acceleration: float = 10.0

    max_rotvel: float = 1.0

    # raycaster_sensor: str = MISSING # currently only feed forward to check if everything works as wanted

    obstacle_center_height: float = 1.0

    # terrain_analysis: TerrainAnalysisCfg = TerrainAnalysisCfg()

    