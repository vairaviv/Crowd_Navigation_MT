

# play statobs env, replace load_run and checkpoint
# ./isaaclab.sh -p ./crowd_navigation_mt/scripts/rsl_rl/play.py --task Isaac-CrowdNavigation-Teacher-StatObs-Anymal-D-v0 --num_envs 4 --logger tensorboard --load_run 2024-11-06_09-50-55_stat_obs_ppo_base_test_2 --checkpoint model_600.pt --device cuda

# play statoby env, conv_nogru
# ./isaaclab.sh -p ./crowd_navigation_mt/scripts/rsl_rl/play.py --task Isaac-CrowdNavigation-Teacher-StatObs-Conv_NoGru-Anymal-D-v0 --num_envs 4 --logger tensorboard --load_run 2024-11-11_17-34-12_statobs_ppo_conv_no_gru --checkpoint model_600.pt --device cuda

# play flat env
# ./isaaclab.sh -p ./crowd_navigation_mt/scripts/rsl_rl/play.py --task Isaac-CrowdNavigation-Teacher-Flat-Beta-Anymal-D-v0 --num_envs 4 --logger tensorboard --load_run 2024-11-12_14-05-35_flat_ppo_base_beta --checkpoint model_100.pt --device cuda

# play a model copied from cluster
# ./isaaclab.sh -p ./crowd_navigation_mt/scripts/rsl_rl/play_cluster_model.py --task Isaac-CrowdNavigation-Teacher-Flat-Beta-Anymal-D-v0 --num_envs 4 --logger tensorboard --load_run 2024-11-13_09-03-35_Flat_PPO_Beta_goal_dist_curr  --checkpoint model_900.pt --device cuda

# play StatObs Conv NoGru 
# ./isaaclab.sh -p ./crowd_navigation_mt/scripts/rsl_rl/play_cluster_model.py --task Isaac-CrowdNavigation-Teacher-StatObs-Conv_NoGru-Anymal-D-v0 --num_envs 4 --logger tensorboard --load_run 2024-11-13_15-49-49_StatObs_lidar_dist_min_rew_Conv_NoGru --checkpoint model_900.pt --device cuda

# Resume Training StatObs Conv NoGru
# ./isaaclab.sh -p ./crowd_navigation_mt/scripts/rsl_rl/train.py --task Isaac-CrowdNavigation-Teacher-StatObs-Conv_NoGru-Anymal-D-v0 --num_envs 4 --logger tensorboard --load_run 2024-11-13_15-49-49_StatObs_lidar_dist_min_rew_Conv_NoGru --checkpoint model_900.pt --device cuda