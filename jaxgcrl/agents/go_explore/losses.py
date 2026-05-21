from typing import Any, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from .types import TrainingState, Transition


def flatten_sac_critic_params(critic_states) -> Dict[str, Any]:
    """Flatten per-critic TrainStates into the ``critic_{i}_{layer}`` layout.

    SAC-style critics (the main SAC Q, the exploration Q) hold one TrainState
    per ensemble member with ``{layer: params}``; this is the shape the
    ensemble network expects at ``apply`` time.
    """
    return {
        f"critic_{i}_{lname}": lparams
        for i, cs in enumerate(critic_states)
        for lname, lparams in cs.params.items()
    }


def flatten_crl_critic_params(critic_states) -> Dict[str, Any]:
    """Flatten per-critic TrainStates into the CRL ``sa_encoder_{i}``/``g_encoder_{i}`` layout."""
    full = {}
    for i, cs in enumerate(critic_states):
        full[f"sa_encoder_{i}"] = cs.params["sa_encoder"]
        full[f"g_encoder_{i}"] = cs.params["g_encoder"]
    return full


def soft_update_target_params(target_params, critic_states, tau: float):
    """Polyak-average a target critic-params tree toward the current ensemble."""
    live = flatten_sac_critic_params(critic_states)
    return jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau, target_params, live,
    )


def energy_fn(name, x, y):
    if name == "norm":
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)
    elif name == "dot":
        return jnp.sum(x * y, axis=-1)
    elif name == "cosine":
        return jnp.sum(x * y, axis=-1) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-6)
    elif name == "l2":
        return -jnp.sum((x - y) ** 2, axis=-1)
    else:
        raise ValueError(f"Unknown energy function: {name}")


def contrastive_loss_fn(name, logits):
    if name == "fwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
    elif name == "bwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=0))
    elif name == "sym_infonce":
        critic_loss = -jnp.mean(
            2 * jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1) - jax.nn.logsumexp(logits, axis=0)
        )
    elif name == "binary_nce":
        critic_loss = -jnp.mean(jax.nn.sigmoid(logits))
    else:
        raise ValueError(f"Unknown contrastive loss function: {name}")
    return critic_loss


def update_actor_and_alpha(config: Dict[str, Any], networks: Dict[str, Any],
                           transitions: Transition, training_state: TrainingState, key: jnp.ndarray):
    """CRL actor and alpha update."""
    def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
        obs = transitions.observation  # expected_shape = self.batch_size, obs_size + goal_size
        state = obs[:, : config["state_size"]]
        future_state = transitions.extras["future_state"]
        goal = future_state[:, config["goal_indices"]]
        observation = jnp.concatenate([state, goal], axis=1)

        # Use actor API
        means, log_stds = networks["actor"].apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        # Split key before stochastic operation
        key, noise_key = jax.random.split(key)
        x_ts = means + stds * jax.random.normal(noise_key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob = log_prob.sum(-1)  # dimension = B

        # Use critic API to compute Q-value - construct obs from state and goal
        critic = networks["critic"]
        obs_with_goal = jnp.concatenate([state, goal], axis=1)
        # critic_params is already the full reconstructed params structure
        q_values = critic.apply(critic_params, obs_with_goal, action)  # Shape: (batch_size, n_critics)
        # Use first critic's output for CRL
        qf_pi = q_values[:, 0]  # Shape: (batch_size,)

        actor_loss_val = jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi)

        return actor_loss_val, log_prob

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - config["target_entropy"]))
        return jnp.mean(alpha_loss)

    full_critic_params = flatten_crl_critic_params(training_state.critic_states)

    (actor_loss_val, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
        training_state.actor_state.params,
        full_critic_params,
        training_state.alpha_state.params["log_alpha"],
        transitions,
        key,
    )
    new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

    alpha_loss_val, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
    new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

    training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

    metrics = {
        "entropy": -jnp.mean(log_prob),  # log_prob: (batch_size,), mean to scalar for consistency
        "actor_loss": actor_loss_val,
        "alpha_loss": alpha_loss_val,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
    }

    return training_state, metrics


def update_critic(config: Dict[str, Any], networks: Dict[str, Any],
                  transitions: Transition, training_state: TrainingState, key: jnp.ndarray):
    """CRL critic update with support for multiple critics.
    
    All critics are updated through a single backprop pass, but each maintains
    its own TrainState and optimizer state (decoupled).
    """
    state = transitions.observation[:, : config["state_size"]]
    action = transitions.action
    goal = transitions.observation[:, config["state_size"] :]

    critic = networks["critic"]
    n_critics = critic.n_critics
    
    # Collect all critic parameters into a tuple for single backprop
    all_critic_params = tuple(critic_i_state.params for critic_i_state in training_state.critic_states)
    
    # Loss function that computes loss for all critics in one pass
    def all_critics_loss(critic_params_tuple, transitions, key):
        """Compute loss for all critics simultaneously."""
        sa_input = jnp.concatenate([state, action], axis=-1)
        
        # Compute loss for each critic
        losses = []
        logsumexps = []
        corrects = []
        logits_pos_list = []
        logits_neg_list = []
        
        for i, critic_i_params in enumerate(critic_params_tuple):
            # Get representations for this critic
            sa_repr = critic.sa_encoders[i].apply(critic_i_params["sa_encoder"], sa_input)
            g_repr = critic.g_encoders[i].apply(critic_i_params["g_encoder"], goal)

            # InfoNCE
            logits = energy_fn(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])
            loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)

            # logsumexp regularisation
            logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
            loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp**2)

            I = jnp.eye(logits.shape[0])
            correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
            logits_pos = jnp.sum(logits * I) / jnp.sum(I)
            logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
            
            losses.append(loss)
            logsumexps.append(logsumexp)
            corrects.append(correct)
            logits_pos_list.append(logits_pos)
            logits_neg_list.append(logits_neg)
        
        # Return total loss (sum of all critic losses) and metrics
        total_loss = sum(losses)
        metrics = (logsumexps, corrects, logits_pos_list, logits_neg_list, losses)
        return total_loss, metrics
    
    # Compute gradients for all critics in one backprop pass
    (total_loss, (logsumexps, corrects, logits_pos_list, logits_neg_list, critic_losses)), all_grads = jax.value_and_grad(
        all_critics_loss, has_aux=True
    )(all_critic_params, transitions, key)
    
    # Apply gradients to each critic's TrainState separately (preserves optimizer state)
    new_critic_states = []
    for i, (critic_i_state, grad) in enumerate(zip(training_state.critic_states, all_grads)):
        new_critic_i_state = critic_i_state.apply_gradients(grads=grad)
        new_critic_states.append(new_critic_i_state)
    
    # Update training state with all updated critic states
    training_state = training_state.replace(critic_states=tuple(new_critic_states))

    # Average metrics for logging
    metrics = {
        "categorical_accuracy": jnp.mean(jnp.array([jnp.mean(c) for c in corrects])),
        "logits_pos": jnp.mean(jnp.array(logits_pos_list)),
        "logits_neg": jnp.mean(jnp.array(logits_neg_list)),
        "logsumexp": jnp.mean(jnp.array([ls.mean() for ls in logsumexps])),
        "critic_loss": jnp.mean(jnp.array(critic_losses)),
    }

    return training_state, metrics


def update_alpha_sac(config: Dict[str, Any], networks: Dict[str, Any],
                     transitions: Transition, training_state: TrainingState, key: jnp.ndarray):
    """SAC alpha update (matching original SAC - updates alpha first, before critic/actor).
    
    Note: Original SAC uses OLD alpha value for critic/actor updates (line 370),
    so we update alpha but critic/actor will use the old value from training_state.
    """
    def alpha_loss(alpha_params, actor_params, transitions, key):
        # Sample actions from current policy to get log_probs
        obs = transitions.observation
        # Use actor API
        means, log_stds = networks["actor"].apply(actor_params, obs)
        stds = jnp.exp(log_stds)
        # Split key before stochastic operation
        key, noise_key = jax.random.split(key)
        x_ts = means + stds * jax.random.normal(noise_key, shape=means.shape, dtype=means.dtype)
        actions = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(actions)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdims=True)
        
        # Alpha loss: -alpha * (log_prob + target_entropy)
        # Original SAC: alpha_loss = -alpha * mean(log_prob + target_entropy)
        # CRITICAL: stop_gradient prevents alpha update from affecting actor params
        alpha = jnp.exp(alpha_params["log_alpha"])
        per_sample = jax.lax.stop_gradient(log_prob + config["target_entropy"])  # (batch, 1)
        sample_weights = config.get("sample_weights", None)
        if sample_weights is None:
            alpha_loss = -alpha * jnp.mean(per_sample)
        else:
            w = sample_weights[:, None]
            alpha_loss = -alpha * jnp.sum(w * per_sample) / (jnp.sum(w) + 1e-8)
        return alpha_loss
    
    alpha_loss_val, alpha_grad = jax.value_and_grad(alpha_loss)(
        training_state.alpha_state.params,
        training_state.actor_state.params,
        transitions,
        key,
    )
    new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)
    training_state = training_state.replace(alpha_state=new_alpha_state)
    
    metrics = {
        "alpha_loss": alpha_loss_val,
        "alpha": jnp.exp(new_alpha_state.params["log_alpha"]),
    }
    
    return training_state, metrics


def update_actor_sac(config: Dict[str, Any], networks: Dict[str, Any],
                     transitions: Transition, training_state: TrainingState, key: jnp.ndarray):
    """SAC actor update."""
    def actor_loss(actor_params, q_params, alpha, transitions, key):
        obs = transitions.observation
        # Use actor API
        means, log_stds = networks["actor"].apply(actor_params, obs)
        stds = jnp.exp(log_stds)
        # Split key before stochastic operation
        key, noise_key = jax.random.split(key)
        x_ts = means + stds * jax.random.normal(noise_key, shape=means.shape, dtype=means.dtype)
        actions = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(actions)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdims=True)

        # Use critic API to get Q-values
        critic = networks["critic"]
        q_values = critic.apply(q_params, obs, actions)
        if config.get("use_sac_critic_mean", False):
            q_value = jnp.mean(q_values, axis=-1, keepdims=True)
        else:
            q_value = jnp.min(q_values, axis=-1, keepdims=True)  # Min over critics

        sample_weights = config.get("sample_weights", None)
        if sample_weights is None:
            actor_loss = jnp.mean(alpha * log_prob - q_value)
        else:
            w = sample_weights[:, None]
            actor_loss = jnp.sum(w * (alpha * log_prob - q_value)) / (jnp.sum(w) + 1e-8)
        return actor_loss, log_prob

    # Use OLD alpha value (before alpha update) - matching original SAC
    alpha = jnp.exp(training_state.alpha_state.params["log_alpha"]) if training_state.alpha_state else 0.0

    full_critic_params = flatten_sac_critic_params(training_state.critic_states)

    (actor_loss_val, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
        training_state.actor_state.params,
        full_critic_params,
        alpha,
        transitions,
        key,
    )
    new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)
    training_state = training_state.replace(actor_state=new_actor_state)

    metrics = {
        "entropy": -jnp.mean(log_prob),
        "actor_loss": actor_loss_val,
    }

    return training_state, metrics


def update_critic_sac(config: Dict[str, Any], networks: Dict[str, Any],
                      transitions: Transition, training_state: TrainingState, key: jnp.ndarray):
    """SAC critic update for an ensemble of critics.

    All critics are updated through a single ``critic.apply`` forward pass and
    one ``value_and_grad`` over the per-critic params tuple, so the cost scales
    O(n_critics) instead of the previous O(n_critics^2). Each critic still
    keeps its own TrainState/optimizer state.
    """
    obs = transitions.observation
    next_obs = transitions.next_observation
    actions = transitions.action
    rewards = transitions.reward
    discounts = transitions.discount

    critic = networks["critic"]
    n_critics = critic.n_critics

    # Use OLD alpha value (before alpha update) - matching original SAC
    alpha = jnp.exp(training_state.alpha_state.params["log_alpha"]) if training_state.alpha_state else 0.0

    # target_critic_params already has the full critic_{i}_{layer} layout.
    target_q_params = training_state.target_critic_params

    # Sample next actions (shared across critics).
    actor = networks["actor"]
    next_means, next_log_stds = actor.apply(training_state.actor_state.params, next_obs)
    next_stds = jnp.exp(next_log_stds)
    key, noise_key = jax.random.split(key)
    next_x_ts = next_means + next_stds * jax.random.normal(noise_key, shape=next_means.shape, dtype=next_means.dtype)
    next_actions = nn.tanh(next_x_ts)
    next_log_prob = jax.scipy.stats.norm.logpdf(next_x_ts, loc=next_means, scale=next_stds)
    next_log_prob -= jnp.log((1 - jnp.square(next_actions)) + 1e-6)
    next_log_prob = next_log_prob.sum(-1, keepdims=True)

    target_q_values = critic.apply(target_q_params, next_obs, next_actions)
    if config.get("use_sac_critic_mean", False):
        target_q_value = jnp.mean(target_q_values, axis=-1, keepdims=True)
    else:
        target_q_value = jnp.min(target_q_values, axis=-1, keepdims=True)
    target = rewards[:, None] + config["discounting"] * discounts[:, None] * (
        target_q_value - alpha * next_log_prob
    )  # (batch_size, 1)

    sample_weights = config.get("sample_weights", None)
    weight_vec = None if sample_weights is None else sample_weights[:, None]

    # Tuple of per-critic params; gradient flows to each through the
    # corresponding column of the stacked Q output.
    critic_params_tuple = tuple(cs.params for cs in training_state.critic_states)

    def all_critics_loss(params_tuple):
        full_params = {
            f"critic_{i}_{lname}": lparams
            for i, p in enumerate(params_tuple)
            for lname, lparams in p.items()
        }
        q_values = critic.apply(full_params, obs, actions)  # (batch_size, n_critics)
        sq_err = (q_values - target) ** 2  # broadcast target across critics
        if weight_vec is None:
            per_critic_loss = jnp.mean(sq_err, axis=0)  # (n_critics,)
        else:
            per_critic_loss = jnp.sum(weight_vec * sq_err, axis=0) / (jnp.sum(weight_vec) + 1e-8)
        return jnp.sum(per_critic_loss), per_critic_loss

    (_, per_critic_loss), grads_tuple = jax.value_and_grad(all_critics_loss, has_aux=True)(
        critic_params_tuple
    )

    new_critic_states = tuple(
        cs.apply_gradients(grads=g)
        for cs, g in zip(training_state.critic_states, grads_tuple)
    )
    training_state = training_state.replace(critic_states=new_critic_states)

    return training_state, {"critic_loss": jnp.mean(per_critic_loss)}
