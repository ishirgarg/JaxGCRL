#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python run.py go-explore-crl \
  --env ant_u_maze_hard \
  --num_goal_conditioned_steps 250 \
  --num_exploratory_steps 250 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.01 \
  --ep_goal_proposer_name env_goals \
  --candidate_goals_type any \
  --use_same_policy \
  --max_replay_size 20000 \
  --total_env_steps 120000000

#   CUDA_VISIBLE_DEVICES=1 python run.py go-explore-crl \
#   --env ant_u_maze \
#   --num_goal_conditioned_steps 900 \
#   --num_exploratory_steps 600 \
#   --gcp_goal_proposer_name maxwaypointratio_one_env \
#   --goal_sampling_temperature 0.01 \
#   --ep_goal_proposer_name env_goals \
#   --candidate_goals_type any \
#   --use_same_policy \
#   --max_replay_size 20000 \
#   --total_env_steps 120000000

# CUDA_VISIBLE_DEVICES=1 python run.py go-explore-crl \
#   --env ant_u_maze \
#   --num_goal_conditioned_steps 900 \
#   --num_exploratory_steps 600 \
#   --gcp_goal_proposer_name mega \
#   --goal_sampling_temperature 0.1 \
#   --ep_goal_proposer_name env_goals \
#   --candidate_goals_type any \
#   --use_same_policy \
#   --max_replay_size 20000 \
#   --total_env_steps 120000000

# CUDA_VISIBLE_DEVICES=1 python run.py go-explore-crl \
#   --env ant_u_maze \
#   --num_goal_conditioned_steps 900 \
#   --num_exploratory_steps 600 \
#   --gcp_goal_proposer_name omega \
#   --ep_goal_proposer_name env_goals \
#   --goal_sampling_temperature 0.1 \
#   --candidate_goals_type any \
#   --use_same_policy \
#   --max_replay_size 20000 \
#   --total_env_steps 120000000
