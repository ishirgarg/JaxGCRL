 #!/bin/bash

CUDA_VISIBLE_DEVICES=3 python run.py go-explore-sac \
  --env ant_u_maze \
  --num_goal_conditioned_steps 900 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --ep_reward_fn state_goal_mi \
  --goal_sampling_temperature 0.1 \
  --candidate_goals_type any \
  --max_replay_size 20000 \
  --total_env_steps 100000000

CUDA_VISIBLE_DEVICES=3 python run.py go-explore-sac \
  --env ant_big_maze \
  --num_goal_conditioned_steps 1200 \
  --num_exploratory_steps 800 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --ep_reward_fn state_goal_mi \
  --goal_sampling_temperature 0.1 \
  --candidate_goals_type any \
  --max_replay_size 20000 \
  --total_env_steps 100000000

CUDA_VISIBLE_DEVICES=3 python run.py go-explore-sac \
  --env ant_u_maze \
  --num_goal_conditioned_steps 900 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name mega \
  --ep_reward_fn state_goal_mi \
  --candidate_goals_type any \
  --max_replay_size 20000 \
  --total_env_steps 100000000

CUDA_VISIBLE_DEVICES=3 python run.py go-explore-sac \
  --env ant_big_maze \
  --num_goal_conditioned_steps 1200 \
  --num_exploratory_steps 800 \
  --gcp_goal_proposer_name mega \
  --ep_reward_fn state_goal_mi \
  --candidate_goals_type any \
  --max_replay_size 20000 \
  --total_env_steps 100000000
