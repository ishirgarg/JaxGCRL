 #!/bin/bash

CUDA_VISIBLE_DEVICES=7 python run.py go-explore-crl \
  --env ant_big_maze \
  --num_goal_conditioned_steps 1200 \
  --num_exploratory_steps 800 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.1 \
  --ep_goal_proposer_name env_goals \
  --candidate_goals_type any \
  --use_same_policy \
  --max_replay_size 20000 \
  --total_env_steps 100000000

CUDA_VISIBLE_DEVICES=7 python run.py go-explore-crl \
  --env ant_big_maze \
  --num_goal_conditioned_steps 1200 \
  --num_exploratory_steps 800 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.1 \
  --ep_goal_proposer_name mega \
  --candidate_goals_type any \
  --use_same_policy \
  --max_replay_size 20000 \
  --total_env_steps 100000000

CUDA_VISIBLE_DEVICES=7 python run.py go-explore-crl \
  --env ant_big_maze \
  --num_goal_conditioned_steps 1200 \
  --num_exploratory_steps 800 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --goal_sampling_temperature 0.1 \
  --ep_goal_proposer_name omega \
  --candidate_goals_type any \
 --use_same_policy \
  --max_replay_size 20000 \
  --total_env_steps 100000000