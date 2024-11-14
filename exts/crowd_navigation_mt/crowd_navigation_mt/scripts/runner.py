# script to run multiple tasks after each other

import subprocess

# define the tasks and their names
""" Currently supported tasks are (to see in .../crowd_navigation_lidar/config/anymal_d/__init__):
    - "Isaac-CrowdNavigation-Teacher-StatObs-Anymal-D-v0"
    - "Isaac-CrowdNavigation-Teacher-StatObs-Height-Anymal-D-v0"
    - "Isaac-CrowdNavigation-Teacher-StatObs-NoGru-Anymal-D-v0"
    - "Isaac-CrowdNavigation-Teacher-DynObs-Anymal-D-v0"

    Evaluation Environment
    - "Isaac-CrowdNavigation-Teacher-StatObs-EVAL-Anymal-D-v0"
"""

tasks = ["Isaac-CrowdNavigation-Teacher-StatObs-Anymal-D-v0"]
task_names = ["stat_obs_gru"]

#py_train_path = "python source/standalone/workflows/rsl_rl/train.py"
#py_train_path = "./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py"
py_train_path = "./crowd_navigation_mt/scripts/rsl_rl/train.py"
for task, task_name in zip(tasks, task_names):
    print(f"Running task: {task_name}")
    command = f"{py_train_path} --task {task} --run_name {task_name} --logger wandb --num_envs 4 --log_project_name MT_Crowd_Navigation"
    # Static Obstacles Terrain
    # ./isaaclab.sh -p ./crowd_navigation_mt/scripts/rsl_rl/train.py --task Isaac-CrowdNavigation-Teacher-StatObs-Anymal-D-v0 --run_name "statobs_ppo_base" --logger wandb --num_envs 2048 --log_project_name MT_Crowd_Navigation_StatObs --headless
    # Flat Terrain
    # ./isaaclab.sh -p ./crowd_navigation_mt/scripts/rsl_rl/train.py --task Isaac-CrowdNavigation-Teacher-Flat-Beta-Anymal-D-v0 --run_name "flat_ppo_base_beta" --logger wandb --num_envs 2048 --log_project_name MT_Crowd_Navigation_Flat --headless
    # Static Obstacles PPO CONV No GRU
    # ./isaaclab.sh -p ./crowd_navigation_mt/scripts/rsl_rl/train.py --task Isaac-CrowdNavigation-Teacher-StatObs-Conv_NoGru-Anymal-D-v0 --run_name "statobs_ppo_conv_no_gru" --logger wandb --num_envs 2048 --log_project_name MT_Crowd_Navigation_StatObs_Conv_NoGru --headless
    subprocess.run(command, shell=True)

"""
In order to push the current code to the Euler Cluster and run a job do:

in shell: ./docker/cluster/cluster_interface.sh job --task {Task_Name} --run_name {for_folder} --num_envs {amount_of_envs} --logger {wandb/tensorboard} --log_project_name {project_folder_wandb} --headless

example for static obstacle terrain with ppo and ActorCriticBeta Module
./docker/cluster/cluster_interface.sh job --task Isaac-CrowdNavigation-Teacher-StatObs-PPO_Beta-Anymal-D-v0 --run_name --num_envs 2048 --logger wandb --log_project_name MT_Crowd_Navigation_StatObs_Beta --headless

To copy the logs something, maybe specify which files exactly, to not copy the whole directory:

scp -r vairaviv@euler.ethz.ch:/cluster/home/vairaviv/isaaclab/logs/rsl_rl/crowd_navigation/{log_file_directory} ./logs/{run_name}


To copy log folder from cluster to docker:

cd path_to_IsaacLab
python ./crowd_navigation_mt/exts/crowd_navigation_mt/crowd_navigation_mt/scripts/copy_from_cluster_to_docker.py --directory_name "name_of_run_in_log_directory"

"""
