# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ANYbotics robots.

The following configuration parameters are available:

* :obj:`ANYMAL_D_EXT_BASE_CFG`: The ANYmal-B robot with ANYdrives 4.0

Reference:

* https://github.com/ANYbotics/anymal_d_simple_description

Assets body contains HIPS too.

"""

from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ActuatorNetLSTMCfg, ActuatorNetMLPCfg, DCMotorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.sensors import RayCasterCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

from crowd_navigation_mt import CROWDNAV_DATA_DIR

ANYDRIVE_4_MLP_ACTUATOR_CFG = ActuatorNetMLPCfg(
    # values matched from legged gym
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    network_file=f"{ISAACLAB_ASSETS_DATA_DIR}/ActuatorNets/ANYbotics/anydrive_4_mlp.jit",
    saturation_effort=140.0,
    effort_limit=80.0,  # see anydrive 3 above
    velocity_limit=8.5,
    input_idx=[0, 2, 4],
    input_order="vel_pos",
    vel_scale=0.2,
    pos_scale=5.0,
    torque_scale=60.0,
)
"""Configuration for ANYdrive 4.0 (used on ANYmal-D) with MLP actuator model."""


ANYMAL_D_EXT_BASE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{CROWDNAV_DATA_DIR}/ANYmal_D/anymal_d_ext_base.usd",
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d_minimal.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_4_MLP_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration of ANYmal-D robot using actuator-net."""