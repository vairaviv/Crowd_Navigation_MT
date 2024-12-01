# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.assets import RigidObject

if TYPE_CHECKING:
    from .sfm_obstacle_importer_cfg import SFMObstacleImporterCfg
    from omni.isaac.lab.assets import RigidObjectCfg
    from omni.isaac.lab.envs import ManagerBasedRLEnv


class SFMObstacleImporter(RigidObject):

    obstacle_dict: dict = dict()

    def __init__(self, cfg: SFMObstacleImporterCfg):
        """Initializes the social force model obstacle importer.

        Args:
            cfg: the configuration for the social force model obstacle importer.

        """
        super().__init__(cfg)

        # # store input configuration
        # self._cfg = cfg
        
        # for i in range(self._cfg.num_sfm_obstacles):
        #     print(i)


        for i in range(cfg.num_sfm_obstacles):
            color = tuple([0.5, 0.5, 0.5])

            obstacle_prim_path_name = f"SFM_Obstacle_{i}"
            self.obstacle_dict[obstacle_prim_path_name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"{obstacle_prim_path_name}",
                spawn=sim_utils.CuboidCfg(
                    size=(0.75, 0.75, 2.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=1.0,
                        disable_gravity=False,
                        max_angular_velocity=1.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=75.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 1.0, 1.0)),
                collision_group=-1,
            )

