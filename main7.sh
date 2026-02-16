 #!/bin/bash

CUDA_VISIBLE_DEVICES=1 python run.py go-explore-sac \
  --env ant_trident_maze \
  --num_goal_conditioned_steps 900 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --ep_reward_fn state_goal_mi \
  --goal_sampling_temperature 0.1 \
  --candidate_goals_type any \
  --max_replay_size 20000 \
  --total_env_steps 100000000

CUDA_VISIBLE_DEVICES=1 python run.py go-explore-sac \
  --env ant_trident_maze \
  --num_goal_conditioned_steps 900 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name mega \
  --ep_reward_fn state_goal_mi \
  --candidate_goals_type any \
  --max_replay_size 20000 \
  --total_env_steps 100000000

 CUDA_VISIBLE_DEVICES=1 python run.py go-explore-crl \
  --env ant_trident_maze \
  --num_goal_conditioned_steps 900 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --ep_goal_proposer_name env_goals \
  --goal_sampling_temperature 0.1 \
  --candidate_goals_type any \
  --use_same_policy \
  --max_replay_size 20000 \
  --total_env_steps 100000000

 CUDA_VISIBLE_DEVICES=1 python run.py go-explore-crl \
  --env ant_trident_maze \
  --num_goal_conditioned_steps 900 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name mega \
  --ep_goal_proposer_name env_goals \
  --candidate_goals_type any \
  --use_same_policy \
  --max_replay_size 20000 \
  --total_env_steps 100000000
