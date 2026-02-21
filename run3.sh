#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python run.py go-explore-crl \
  --env ant_trident_maze \
  --num_goal_conditioned_steps 250 \
  --num_exploratory_steps 250 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.01 \
  --ep_goal_proposer_name nearest_env_goal_to_gcp_goal \
  --candidate_goals_type any \
  --use_same_policy \
  --use-gcp-noise \
  --max_replay_size 20000 \
  --total_env_steps 80000000

CUDA_VISIBLE_DEVICES=1 python run.py go-explore-crl \
  --env ant_snake_maze \
  --num_goal_conditioned_steps 1500 \
  --num_exploratory_steps 1500 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.01 \
  --ep_goal_proposer_name nearest_env_goal_to_gcp_goal \
  --candidate_goals_type any \
  --use_same_policy \
  --use-gcp-noise \
  --max_replay_size 20000 \
  --total_env_steps 200000000
