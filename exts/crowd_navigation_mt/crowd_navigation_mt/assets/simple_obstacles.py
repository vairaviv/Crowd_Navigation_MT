# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBase, AssetBaseCfg, RigidObject, RigidObjectCfg 


from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG


##
# Pre-defined configs
##


ROBOT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.CuboidCfg(
        size=(1.5, 1, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=False),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.2)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    collision_group=-1,
)

SENSOR_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Sensor",
    spawn=sim_utils.CuboidCfg(
        size=(0.1, 0.2, 0.2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=False),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, 0.5)),
    collision_group=0,
)

CYLINDER_HUMANOID_CFG = RigidObjectCfg(
    spawn=sim_utils.CylinderCfg(
        radius=0.5,
        height=2,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=None,  # sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.9, 0.6)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 1.0, 1.0)),
    collision_group=-1,
)

OBS_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        size=(0.75, 0.75, 2.0),  # (0.3, 0.75, 2.0)
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=None,  # sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(metallic=0.2, diffuse_color=(0.6, 0.3, 0.0)),
    ),
    collision_group=-1,
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 1.0, 1), lin_vel=(1.0, 0.0, 0.0)),
)

EMPTY_OBS_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        size=(0.05, 0.05, 0.001),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(metallic=0.2, diffuse_color=(0.1, 0.1, 0.1)),
    ),
    collision_group=-1,
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 1.0, 1), lin_vel=(1.0, 0.0, 0.0)),
)

OBS_CFG_WORKING = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        size=(1.0, 0.5, 0.7),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=None,  # sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(metallic=0.2),
    ),
    collision_group=-1,
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.6), lin_vel=(1.0, 0.0, 0.0)),
)

WALL_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Obstacle",
    spawn=sim_utils.CuboidCfg(
        size=(50, 0.5, 2),
        # rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=False),
        # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 10.0, 1.0)),
    collision_group=-1,
)


# markers:
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkersCfg

MY_RAY_CASTER_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "ground_hits": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        "obstacle_hits": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)
MY_RAY_CASTER_BLUE_CFG = VisualizationMarkersCfg(
    markers={
        "ground_hits": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "obstacle_hits": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    },
)

