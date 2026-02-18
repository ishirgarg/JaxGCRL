#!/bin/bash
 
CUDA_VISIBLE_DEVICES=0 python run.py go-explore-sac --env ant_u_maze --num_goal_conditioned_steps 800 --num_exploratory_steps 100 --gcp_goal_proposer_name maxwaypointratio_one_env --ep_reward_fn state_goal_mi --goal_sampling_temperature 0.01 --candidate_goals_type any --max_replay_size 20000 --total_env_steps 100000000
