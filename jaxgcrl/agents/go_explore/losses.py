from typing import Any, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from .types import TrainingState, Transition


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

        means, log_stds = networks["actor"].apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
        log_prob = log_prob.sum(-1)  # dimension = B

        sa_encoder_params, g_encoder_params = (
            critic_params["sa_encoder"],
            critic_params["g_encoder"],
        )
        sa_repr = networks["sa_encoder"].apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
        g_repr = networks["g_encoder"].apply(g_encoder_params, goal)

        qf_pi = energy_fn(config["energy_fn"], sa_repr, g_repr)

        actor_loss = jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi)

        return actor_loss, log_prob

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - config["target_entropy"]))
        return jnp.mean(alpha_loss)

    (actor_loss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
        training_state.actor_state.params,
        training_state.critic_state.params,
        training_state.alpha_state.params["log_alpha"],
        transitions,
        key,
    )
    new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
    new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

    training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

    metrics = {
        "entropy": -log_prob,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
    }

    return training_state, metrics


def update_critic(config: Dict[str, Any], networks: Dict[str, Any],
                  transitions: Transition, training_state: TrainingState, key: jnp.ndarray):
    """CRL critic update."""
    def critic_loss(critic_params, transitions, key):
        sa_encoder_params, g_encoder_params = (
            critic_params["sa_encoder"],
            critic_params["g_encoder"],
        )

        state = transitions.observation[:, : config["state_size"]]
        action = transitions.action

        sa_repr = networks["sa_encoder"].apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
        g_repr = networks["g_encoder"].apply(
            g_encoder_params, transitions.observation[:, config["state_size"] :]
        )

        # InfoNCE
        logits = energy_fn(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])
        critic_loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)

        # logsumexp regularisation
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        critic_loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp**2)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return critic_loss, (logsumexp, I, correct, logits_pos, logits_neg)

    (loss, (logsumexp, I, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
        critic_loss, has_aux=True
    )(training_state.critic_state.params, transitions, key)
    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    training_state = training_state.replace(critic_state=new_critic_state)

    metrics = {
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
        "critic_loss": loss,
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
        means, log_stds = networks["actor"].apply(actor_params, obs)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        actions = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(actions)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdims=True)
        
        # Alpha loss: -alpha * (log_prob + target_entropy)
        # Original SAC: alpha_loss = -alpha * mean(log_prob + target_entropy)
        alpha = jnp.exp(alpha_params["log_alpha"])
        alpha_loss = -alpha * jnp.mean(log_prob + config["target_entropy"])
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
        means, log_stds = networks["actor"].apply(actor_params, obs)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        actions = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= jnp.log((1 - jnp.square(actions)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdims=True)

        # Get Q-values for the sampled actions
        critic_network = networks["critic"]
        q_values = critic_network.apply(q_params, obs, actions)
        q_value = jnp.min(q_values, axis=-1, keepdims=True)  # Min over critics

        actor_loss = jnp.mean(alpha * log_prob - q_value)
        return actor_loss, log_prob

    # Use OLD alpha value (before alpha update) - matching original SAC
    alpha = jnp.exp(training_state.alpha_state.params["log_alpha"]) if training_state.alpha_state else 0.0
    
    (actor_loss_val, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
        training_state.actor_state.params,
        training_state.critic_state.params,
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
    """SAC critic update."""
    def critic_loss(q_params, actor_params, target_q_params, alpha, transitions, key):
        obs = transitions.observation
        next_obs = transitions.next_observation
        actions = transitions.action
        rewards = transitions.reward
        discounts = transitions.discount

        # Current Q-values
        critic_network = networks["critic"]
        actor_network = networks["actor"]
        q_values = critic_network.apply(q_params, obs, actions)
        
        # Target Q-values using next state and next action from policy
        next_means, next_log_stds = actor_network.apply(actor_params, next_obs)
        next_stds = jnp.exp(next_log_stds)
        next_x_ts = next_means + next_stds * jax.random.normal(key, shape=next_means.shape, dtype=next_means.dtype)
        next_actions = nn.tanh(next_x_ts)
        next_log_prob = jax.scipy.stats.norm.logpdf(next_x_ts, loc=next_means, scale=next_stds)
        next_log_prob -= jnp.log((1 - jnp.square(next_actions)) + 1e-6)
        next_log_prob = next_log_prob.sum(-1, keepdims=True)

        target_q_values = critic_network.apply(target_q_params, next_obs, next_actions)
        target_q_value = jnp.min(target_q_values, axis=-1, keepdims=True)  # Min over critics
        target = rewards[:, None] + config["discounting"] * discounts[:, None] * (
            target_q_value - alpha * next_log_prob
        )

        # Bellman error for each critic
        critic_loss = jnp.mean((q_values - target[..., None]) ** 2)
        return critic_loss

    # Use OLD alpha value (before alpha update) - matching original SAC (line 370)
    # Original uses training_state.alpha_params (old) even after alpha_update returns new params
    alpha = jnp.exp(training_state.alpha_state.params["log_alpha"]) if training_state.alpha_state else 0.0
    
    # target_critic_params is always set for SAC (initialized in baseline.py line 201)
    # This function is only called for SAC via SACCritic.update
    target_q_params = training_state.target_critic_params
    
    critic_loss_val, critic_grad = jax.value_and_grad(critic_loss)(
        training_state.critic_state.params,
        training_state.actor_state.params,
        target_q_params,
        alpha,
        transitions,
        key,
    )
    new_critic_state = training_state.critic_state.apply_gradients(grads=critic_grad)
    training_state = training_state.replace(critic_state=new_critic_state)

    metrics = {
        "critic_loss": critic_loss_val,
    }

    return training_state, metrics
