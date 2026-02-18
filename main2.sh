#!/bin/bash

<<<<<<< Updated upstream

# ant_u_maze — maxwaypointratio_one_env (temp 0.1) — shared policy
CUDA_VISIBLE_DEVICES=1 python run.py go-explore-crl \
  --env ant_u_maze \
  --num_goal_conditioned_steps 400 \
  --num_exploratory_steps 400 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.1 \
  --ep_goal_proposer_name env_goals \
  --use_same_policy \
  --total_env_steps 150000000

# ant_u_maze — maxwaypointratio_one_env (temp 0.1) — shared policy
CUDA_VISIBLE_DEVICES=1 python run.py go-explore-crl \
  --env ant_u_maze \
  --num_goal_conditioned_steps 400 \
  --num_exploratory_steps 400 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.1 \
  --ep_goal_proposer_name env_goals \
  --total_env_steps 150000000

  CUDA_VISIBLE_DEVICES=1 python run.py go-explore-crl \
  --env ant_u_maze \
  --num_goal_conditioned_steps 400 \
  --num_exploratory_steps 400 \
=======
CUDA_VISIBLE_DEVICES=0 python run.py go-explore-crl \
  --env ant_u_maze_hard \
  --num_goal_conditioned_steps 900 \
  --num_exploratory_steps 600 \
>>>>>>> Stashed changes
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.1 \
  --ep_goal_proposer_name env_goals \
<<<<<<< Updated upstream
  --use_same_policy \
  --train_ep_on_main_buffer \
  --total_env_steps 150000000

# ant_u_maze — maxwaypointratio_one_env (temp 0.1) — shared policy
CUDA_VISIBLE_DEVICES=1 python run.py go-explore-crl \
  --env ant_u_maze \
  --num_goal_conditioned_steps 400 \
  --num_exploratory_steps 400 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.1 \
  --ep_goal_proposer_name env_goals \
  --train_ep_on_main_buffer \
  --total_env_steps 150000000


## ant_big_maze — maxwaypointratio_one_env (temp 0.1) — shared policy
CUDA_VISIBLE_DEVICES=1 python run.py go-explore-crl \
  --env ant_big_maze \
  --num_goal_conditioned_steps 500 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.1 \
  --ep_goal_proposer_name env_goals \
  --total_env_steps 200000000


# ant_big_maze — maxwaypointratio_one_env (temp 0.1) — shared policy
CUDA_VISIBLE_DEVICES=1 python run.py go-explore-crl \
  --env ant_big_maze \
  --num_goal_conditioned_steps 500 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.1 \
  --ep_goal_proposer_name env_goals \
  --use_same_policy \
  --total_env_steps 200000000




# ant_u_maze — maxwaypointratio_one_env (temp 0.1) — shared policy
CUDA_VISIBLE_DEVICES=1 python run.py go-explore-crl \
  --env ant_u_maze \
  --num_goal_conditioned_steps 400 \
  --num_exploratory_steps 400 \
  --gcp_goal_proposer_name ucgr \
  --ep_goal_proposer_name env_goals \
  --total_env_steps 150000000


# ant_u_maze — maxwaypointratio_one_env (temp 0.1) — shared policy
CUDA_VISIBLE_DEVICES=1 python run.py go-explore-crl \
  --env ant_u_maze \
  --num_goal_conditioned_steps 400 \
  --num_exploratory_steps 400 \
  --gcp_goal_proposer_name ucgr \
  --ep_goal_proposer_name env_goals \
  --use_same_policy \
  --total_env_steps 150000000
=======
  --candidate_goals_type any \
  --max_replay_size 20000 \
  --total_env_steps 120000000
>>>>>>> Stashed changes
