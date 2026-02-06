#!/bin/bash

# ant_u_maze — ucgr — shared policy
python run.py go-explore-crl \
  --env ant_u_maze \
  --num_goal_conditioned_steps 400 \
  --num_exploratory_steps 400 \
  --gcp_goal_proposer_name ucgr \
  --ep_goal_proposer_name env_goals \
  --use_same_policy \
  --total_env_steps 120000000


# ant_u_maze — maxwaypointratio_one_env (temp 0.1) — shared policy
python run.py go-explore-crl \
  --env ant_u_maze \
  --num_goal_conditioned_steps 400 \
  --num_exploratory_steps 400 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.1 \
  --ep_goal_proposer_name env_goals \
  --use_same_policy \
  --total_env_steps 120000000


# ant_big_maze — ucgr — shared policy
python run.py go-explore-crl \
  --env ant_big_maze \
  --num_goal_conditioned_steps 500 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name ucgr \
  --ep_goal_proposer_name env_goals \
  --use_same_policy \
  --total_env_steps 180000000


# ant_big_maze — maxwaypointratio_one_env (temp 0.1) — shared policy
python run.py go-explore-crl \
  --env ant_big_maze \
  --num_goal_conditioned_steps 500 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.1 \
  --ep_goal_proposer_name env_goals \
  --use_same_policy \
  --total_env_steps 180000000
