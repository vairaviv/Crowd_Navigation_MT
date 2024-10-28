import os

CROWDNAV_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

CROWDNAV_DATA_DIR = os.path.join(CROWDNAV_EXT_DIR, "data")


# This registers the Gym environments via the __init__.py file in the agent_config directory.
from .agent_config import *  # noqa: F401, F403

# Environment configurations
# 
# from .env_config import *  # noqa: F401, F403

from .env_config.env_cfg_base import NavigationTemplateEnvCfg_DEV, NavigationTemplateEnvCfg_TRAIN, NavigationTemplateEnvCfg_PLAY
from .env_config import helper_configurations