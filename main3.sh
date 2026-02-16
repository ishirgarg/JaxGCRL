#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python run.py go-explore-crl \
  --env ant_big_maze \
  --num_goal_conditioned_steps 1200 \
  --num_exploratory_steps 800 \
  --gcp_goal_proposer_name mega \
  --ep_goal_proposer_name env_goals \
  --candidate_goals_type any \
  --use_same_policy \
  --max_replay_size 20000 \
  --total_env_steps 160000000

CUDA_VISIBLE_DEVICES=2 python run.py go-explore-crl \
  --env ant_big_maze \
  --num_goal_conditioned_steps 1200 \
  --num_exploratory_steps 800 \
  --gcp_goal_proposer_name omega \
  --ep_goal_proposer_name env_goals \
  --candidate_goals_type any \
  --use_same_policy \
  --max_replay_size 20000 \
  --total_env_steps 160000000