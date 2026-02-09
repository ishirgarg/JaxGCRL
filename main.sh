#!/bin/bash

## ant_big_maze — maxwaypointratio_one_env (temp 0.1) — shared policy
CUDA_VISIBLE_DEVICES=2 python run.py go-explore-crl \
  --env ant_big_maze \
  --num_goal_conditioned_steps 500 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name ucgr \
  --ep_goal_proposer_name env_goals \
  --total_env_steps 200000000


# ant_big_maze — maxwaypointratio_one_env (temp 0.1) — shared policy
CUDA_VISIBLE_DEVICES=2 python run.py go-explore-crl \
  --env ant_big_maze \
  --num_goal_conditioned_steps 500 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name ucgr \
  --ep_goal_proposer_name env_goals \
  --use_same_policy \
  --total_env_steps 200000000



# ant_big_maze — maxwaypointratio_one_env (temp 0.1) — shared policy
CUDA_VISIBLE_DEVICES=2 python run.py go-explore-crl \
  --env ant_big_maze_one_corner \
  --num_goal_conditioned_steps 500 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.1 \
  --ep_goal_proposer_name env_goals \
  --use_same_policy \
  --total_env_steps 250000000

# ant_big_maze — maxwaypointratio_one_env (temp 0.1) — shared policy
CUDA_VISIBLE_DEVICES=2 python run.py go-explore-crl \
  --env ant_big_maze_two_corner \
  --num_goal_conditioned_steps 500 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.1 \
  --ep_goal_proposer_name env_goals \
  --use_same_policy \
  --total_env_steps 250000000

  # ant_big_maze — maxwaypointratio_one_env (temp 0.1) — shared policy
CUDA_VISIBLE_DEVICES=2 python run.py go-explore-crl \
  --env ant_big_maze_three_corner \
  --num_goal_conditioned_steps 500 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.1 \
  --ep_goal_proposer_name env_goals \
  --use_same_policy \
  --total_env_steps 250000000