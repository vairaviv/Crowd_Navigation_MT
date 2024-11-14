import os
import sys

CROWDNAV_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

CROWDNAV_DATA_DIR = os.path.join(CROWDNAV_EXT_DIR, "data")

# # Adjust the path to where the submodule is located
# rsl_rl_submodule_path = os.path.join(os.path.dirname(__file__), '../../../rsl_rl')

# # Insert the submodule path at the start of sys.path
# sys.path.insert(0, rsl_rl_submodule_path)


# This registers the Gym environments via the __init__.py file in the agent_config directory.
from .agent_config import *  # noqa: F401, F403

# Environment configurations
# 
# from .env_config import *  # noqa: F401, F403

from .env_config.env_cfg_base import NavigationTemplateEnvCfg_DEV, NavigationTemplateEnvCfg_TRAIN, NavigationTemplateEnvCfg_PLAY
from .env_config import helper_configurations
# from ....rsl_rl import *
# from ....rsl_rl.rsl_rl.runners import OnPolicyCurriculumRunner, OnPolicyRunner