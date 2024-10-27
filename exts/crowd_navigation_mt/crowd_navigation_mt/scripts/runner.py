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
py_train_path = "./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py"
for task, task_name in zip(tasks, task_names):
    print(f"Running task: {task_name}")
    command = f"{py_train_path} --task {task} --run_name {task_name} --logger wandb --num_envs 4 --log_project_name MT_Crowd_Navigation"
    subprocess.run(command, shell=True)
