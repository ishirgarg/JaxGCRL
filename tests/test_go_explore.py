import jax
import jax.numpy as jnp
import pytest
from brax.envs.base import Env, State

from jaxgcrl.agents.go_explore.goal_proposers import (
    _jax_gaussian_kde,
    _sample_idx_from_temperature_logits,
    create_empowerment_goal_proposer,
    create_mega_goal_proposer,
)
from jaxgcrl.agents.go_explore.types import GoalProposerState, Transition
from jaxgcrl.envs.wrappers import EpisodeWrapper, GoExploreWrapper, VmapWrapper

STATE_DIM = 4
GOAL_SIZE = 2
ACT = 2


class _SyntheticEnv(Env):
    def __init__(self, success_value=0.0):
        self._success = float(success_value)

    def reset(self, rng, goal=None, start=None):
        st = jnp.zeros(STATE_DIM)
        if start is not None:
            st = st.at[jnp.array([0, 1])].set(start)
        g = jnp.zeros(GOAL_SIZE) if goal is None else goal
        obs = jnp.concatenate([st, g])
        metrics = {"success": jnp.full((), self._success), "dist": jnp.ones(())}
        return State(pipeline_state=jnp.zeros(3), obs=obs, reward=jnp.zeros(()),
                     done=jnp.zeros(()), metrics=metrics, info={})

    def step(self, state, action):
        metrics = {"success": jnp.full((), self._success), "dist": jnp.ones(())}
        return state.replace(reward=jnp.zeros(()), done=jnp.zeros(()), metrics=metrics)

    @property
    def observation_size(self):
        return STATE_DIM + GOAL_SIZE

    @property
    def action_size(self):
        return ACT

    @property
    def backend(self):
        return "synthetic"


def _make_go_explore_env(num_gcp_steps=2, num_ep_steps=2, episode_length=100,
                         success_value=0.0):
    env = _SyntheticEnv(success_value=success_value)
    env = VmapWrapper(env)
    env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=1)
    env = GoExploreWrapper(env, num_gcp_steps=num_gcp_steps, num_ep_steps=num_ep_steps,
                           state_size=STATE_DIM,
                           goal_indices=jnp.array([0, 1]))
    return env


def _run_steps(env, num_envs, n_steps, seed=0):
    key = jax.random.PRNGKey(seed)
    rngs = jax.random.split(key, num_envs)
    state = env.reset(rngs, goal=jnp.ones((num_envs, GOAL_SIZE)))
    step = jax.jit(env.step)
    action = jnp.zeros((num_envs, ACT))
    for _ in range(n_steps):
        key, sk = jax.random.split(key)
        state = step(state, action, sk)
    return state


def test_gaussian_kde_shape_and_locality():
    data = jnp.array([[0.0, 0.0], [0.1, 0.1], [0.0, 0.1]])
    query = jnp.array([[0.0, 0.0], [10.0, 10.0]])
    dens = _jax_gaussian_kde(query, data, bandwidth=0.1)
    assert dens.shape == (2,)
    assert jnp.all(dens >= 0)
    assert dens[0] > dens[1]


def test_temperature_sampling_greedy_and_stochastic():
    logits = jnp.array([0.1, 5.0, -2.0, 1.0])
    greedy = _sample_idx_from_temperature_logits(jax.random.PRNGKey(0), logits, 0.0)
    assert int(greedy) == int(jnp.argmax(logits))
    idx = _sample_idx_from_temperature_logits(jax.random.PRNGKey(1), logits, 1.0)
    assert 0 <= int(idx) < logits.shape[0]


def _dummy_proposer_state(num_envs_buf=4, ep_len=8):
    obs_size = STATE_DIM + GOAL_SIZE
    obs = jax.random.uniform(jax.random.PRNGKey(3), (num_envs_buf, ep_len, obs_size))
    trans = Transition(
        observation=obs,
        action=jnp.zeros((num_envs_buf, ep_len, ACT)),
        reward=jnp.zeros((num_envs_buf, ep_len)),
        discount=jnp.ones((num_envs_buf, ep_len)),
        next_observation=obs,
        extras={"state_extras": {"traj_id": jnp.zeros((num_envs_buf, ep_len)),
                                 "truncation": jnp.zeros((num_envs_buf, ep_len))}},
    )
    return GoalProposerState(transitions_sample=trans)


def test_mega_proposer_returns_a_candidate():
    proposer = create_mega_goal_proposer(None, num_envs=4, num_candidates=16,
                                         state_size=STATE_DIM, goal_indices=(0, 1))
    gps = _dummy_proposer_state()
    start_obs = jnp.zeros(STATE_DIM + GOAL_SIZE)
    goal, _, log_data = proposer(jax.random.PRNGKey(0), start_obs, gps)
    assert goal.shape == (GOAL_SIZE,)
    assert bool(jnp.any(jnp.all(log_data["candidate_goals"] == goal, axis=-1)))


def test_empowerment_proposer_uses_scorer():
    def fake_scorer(states, rng, gps):
        return states[:, 0]

    proposer = create_empowerment_goal_proposer(
        None, num_envs=4, num_candidates=16, state_size=STATE_DIM,
        goal_indices=(0, 1), empowerment_scorer=fake_scorer, temperature=0.0, alpha=1.0,
    )
    gps = _dummy_proposer_state()
    goal, _, log_data = proposer(jax.random.PRNGKey(0), jnp.zeros(STATE_DIM + GOAL_SIZE), gps)
    assert goal.shape == (GOAL_SIZE,)
    assert log_data["emp_scores"].shape == (16,)


def test_empowerment_proposer_requires_scorer():
    with pytest.raises(ValueError):
        create_empowerment_goal_proposer(None, 4, 16, STATE_DIM, (0, 1), None)


def test_go_explore_wrapper_deferred_traj_id():
    env = _make_go_explore_env(num_gcp_steps=2, num_ep_steps=2)
    num_envs = 3
    key = jax.random.PRNGKey(0)
    rngs = jax.random.split(key, num_envs)
    state = env.reset(rngs, goal=jnp.ones((num_envs, GOAL_SIZE)))
    step = jax.jit(env.step)
    action = jnp.zeros((num_envs, ACT))

    traj_ids, truncs = [], []
    for _ in range(5):
        key, sk = jax.random.split(key)
        state = step(state, action, sk)
        traj_ids.append(int(state.info["traj_id"][0]))
        truncs.append(int(state.info["truncation"][0]))

    assert traj_ids == [0, 0, 1, 1, 3]
    assert truncs == [0, 1, 0, 1, 0]


def test_go_explore_success_not_counted_when_goal_and_episode_end_coincide():
    env = _make_go_explore_env(num_gcp_steps=10, num_ep_steps=10,
                               episode_length=1, success_value=1.0)
    state = _run_steps(env, num_envs=3, n_steps=5)
    successes = state.info["go_successes_total"]
    completions = state.info["go_completions_total"]
    assert bool(jnp.all(successes <= completions))
    assert bool(jnp.all(successes == 0.0))


def test_go_explore_success_counted_on_normal_goal_reach():
    env = _make_go_explore_env(num_gcp_steps=2, num_ep_steps=2,
                               episode_length=100, success_value=1.0)
    state = _run_steps(env, num_envs=3, n_steps=8)
    successes = state.info["go_successes_total"]
    completions = state.info["go_completions_total"]
    assert bool(jnp.all(successes <= completions))
    assert bool(jnp.all(completions > 0.0))
    assert bool(jnp.all(successes == completions))
