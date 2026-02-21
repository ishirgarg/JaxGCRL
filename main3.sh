#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python run.py go-explore-crl \
  --env ant_big_maze_three_corner \
  --num_goal_conditioned_steps 500 \
  --num_exploratory_steps 500 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.01 \
  --ep_goal_proposer_name env_goals \
  --candidate_goals_type any \
  --use_same_policy \
  --max_replay_size 20000 \
  --total_env_steps 200000000


CUDA_VISIBLE_DEVICES=6 python run.py go-explore-crl \
  --env ant_big_maze_three_corner \
  --num_goal_conditioned_steps 500 \
  --num_exploratory_steps 500 \
  --gcp_goal_proposer_name mega \
  --ep_goal_proposer_name env_goals \
  --candidate_goals_type any \
  --use_same_policy \
  --max_replay_size 20000 \
  --total_env_steps 200000000

CUDA_VISIBLE_DEVICES=6 python run.py go-explore-crl \
  --env ant_big_maze_three_corner \
  --num_goal_conditioned_steps 500 \
  --num_exploratory_steps 500 \
  --gcp_goal_proposer_name omega \
  --ep_goal_proposer_name env_goals \
  --candidate_goals_type any \
  --use_same_policy \
  --max_replay_size 20000 \
  --total_env_steps 200000000
