#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python run.py go-explore-crl \
  --env ant_u_maze \
  --num_goal_conditioned_steps 250 \
  --num_exploratory_steps 250 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.01 \
  --ep_goal_proposer_name nearest_env_goal \
  --candidate_goals_type any \
  --no-use_same_policy \
  --max_replay_size 20000 \
  --total_env_steps 50000000

CUDA_VISIBLE_DEVICES=0 python run.py go-explore-crl \
  --env ant_u_maze \
  --num_goal_conditioned_steps 250 \
  --num_exploratory_steps 250 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.01 \
  --ep_goal_proposer_name nearest_env_goal \
  --candidate_goals_type final \
  --use_same_policy \
  --max_replay_size 20000 \
  --total_env_steps 50000000

CUDA_VISIBLE_DEVICES=0 python run.py go-explore-crl \
  --env ant_u_maze \
  --num_goal_conditioned_steps 250 \
  --num_exploratory_steps 250 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.01 \
  --ep_goal_proposer_name nearest_env_goal \
  --candidate_goals_type any \
  --train_ep_on_main_buffer \
  --no-use_same_policy \
  --max_replay_size 20000 \
  --total_env_steps 50000000