
 CUDA_VISIBLE_DEVICES=2 python run.py go-explore-crl \
  --env ant_u_maze \
  --num_goal_conditioned_steps 900 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --ep_goal_proposer_name env_goals \
  --goal_sampling_temperature 0.01 \
  --use-same-policy \
  --candidate_goals_type any \
  --max_replay_size 20000 \
  --total_env_steps 120000000

 CUDA_VISIBLE_DEVICES=2 python run.py go-explore-crl \
  --env ant_u_maze \
  --num_goal_conditioned_steps 900 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --ep_goal_proposer_name env_goals \
  --goal_sampling_temperature 0.01 \
  --candidate_goals_type any \
  --max_replay_size 20000 \
  --total_env_steps 120000000


 CUDA_VISIBLE_DEVICES=2 python run.py go-explore-crl \
  --env ant_big_maze \
  --num_goal_conditioned_steps 1200 \
  --num_exploratory_steps 800 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --ep_goal_proposer_name env_goals \
  --goal_sampling_temperature 0.01 \
  --use-same-policy \
  --candidate_goals_type any \
  --max_replay_size 20000 \
  --total_env_steps 160000000

 CUDA_VISIBLE_DEVICES=2 python run.py go-explore-crl \
  --env ant_trident_maze \
  --num_goal_conditioned_steps 900 \
  --num_exploratory_steps 600 \
  --gcp_goal_proposer_name maxwaypointratio_one_env \
  --ep_goal_proposer_name env_goals \
  --goal_sampling_temperature 0.01 \
  --use-same-policy \
  --candidate_goals_type any \
  --max_replay_size 20000 \
  --total_env_steps 160000000