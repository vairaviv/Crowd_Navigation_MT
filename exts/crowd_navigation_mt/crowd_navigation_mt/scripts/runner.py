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
    # ./isaaclab.sh -p ./crowd_navigation_mt/scripts/rsl_rl/train.py --task Isaac-CrowdNavigation-Teacher-Flat-Anymal-D-v0 --run_name "flat_ppo_base_goal_heading_clamped" --logger wandb --num_envs 2048 --log_project_name MT_Crowd_Navigation_Flat --headless
    subprocess.run(command, shell=True)

"""
In order to push the current code to the Euler Cluster and run a job do:

in shell: ./docker/cluster/cluster_interface.sh job --task {Task_Name} --run_name {for_folder} --num_envs {amount_of_envs} --logger {wandb/tensorboard} --log_project_name {project_folder_wandb} --headless



To copy the logs something, maybe specify which files exactly, to not copy the whole directory:

scp -r vairaviv@euler.ethz.ch:/cluster/home/vairaviv/isaaclab/logs/rsl_rl/crowd_navigation/{log_file_directory} ./logs/{run_name}


To copy files from local to docker:


"""
