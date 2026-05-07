import flax.linen as nn
import jax
import jax.numpy as jnp


def flatten_q_ensemble_params(critic_states):
    """Flatten per-critic TrainStates into the ``critic_{i}_{layer}`` layout
    expected by the SAC-style ``QNetwork`` ensemble used for the exploration Q.
    """
    return {
        f"critic_{i}_{lname}": lparams
        for i, cs in enumerate(critic_states)
        for lname, lparams in cs.params.items()
    }


def soft_update_q_ensemble_target(target_params, critic_states, tau: float):
    """Polyak-average the exploration-Q target params toward the current ensemble."""
    live = flatten_q_ensemble_params(critic_states)
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


def update_actor_and_alpha(config, networks, transitions, training_state, key):
    """CRL actor + alpha update.

    If ``networks["exploration_q_critic"]`` is provided and the training state
    has ``exploration_q_critic_states`` set, the actor loss also subtracts
    ``min_i Q_exp_i(obs, action)`` so the policy is biased toward actions with
    high exploration Q. The exploration Q is trained on the *weighted* bonus,
    so per-bonus weights are already baked in.
    """
    exploration_q_critic = networks.get("exploration_q_critic")
    exploration_q_states = getattr(
        training_state, "exploration_q_critic_states", None,
    )
    use_exploration_q = (
        exploration_q_critic is not None and exploration_q_states is not None
    )

    def actor_loss(actor_params, critic_params, log_alpha, exp_q_params, transitions, key):
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

        if use_exploration_q:
            exp_q_values = exploration_q_critic.apply(exp_q_params, observation, action)
            exp_q = jnp.min(exp_q_values, axis=-1)
            actor_loss = actor_loss - jnp.mean(exp_q)

        return actor_loss, log_prob

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - config["target_entropy"]))
        return jnp.mean(alpha_loss)

    exp_q_full_params = None
    if use_exploration_q:
        exp_q_full_params = jax.lax.stop_gradient(
            flatten_q_ensemble_params(exploration_q_states)
        )

    (actor_loss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
        training_state.actor_state.params,
        training_state.critic_state.params,
        training_state.alpha_state.params["log_alpha"],
        exp_q_full_params,
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


def update_critic(config, networks, transitions, training_state, key):
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


def update_exploration_q_critic(
    config, networks, transitions, exploration_reward, training_state, key,
):
    """SAC-style Bellman backup of an exploration Q-critic on the bonus reward.

    ``exploration_reward`` is the per-transition weighted bonus (per-bonus
    weight already applied) and replaces the env reward inside the Bellman
    target. The CRL contrastive critic and CRL actor never read reward, so
    overwriting reward here is purely scoped to this call.
    """
    actor = networks["actor"]
    exploration_q_critic = networks["exploration_q_critic"]

    obs = transitions.observation
    next_obs = transitions.next_observation
    actions = transitions.action
    discounts = transitions.discount

    next_means, next_log_stds = actor.apply(
        training_state.actor_state.params, next_obs,
    )
    next_stds = jnp.exp(next_log_stds)
    key, noise_key = jax.random.split(key)
    next_x_ts = next_means + next_stds * jax.random.normal(
        noise_key, shape=next_means.shape, dtype=next_means.dtype,
    )
    next_actions = nn.tanh(next_x_ts)

    target_q_values = exploration_q_critic.apply(
        training_state.exploration_q_target_critic_params, next_obs, next_actions,
    )
    target_q_value = jnp.min(target_q_values, axis=-1, keepdims=True)
    target = exploration_reward[:, None] + config["discounting"] * discounts[:, None] * target_q_value

    critic_states = training_state.exploration_q_critic_states
    critic_params_tuple = tuple(cs.params for cs in critic_states)

    def all_critics_loss(params_tuple):
        full_params = {
            f"critic_{i}_{lname}": lparams
            for i, p in enumerate(params_tuple)
            for lname, lparams in p.items()
        }
        q_values = exploration_q_critic.apply(full_params, obs, actions)
        sq_err = (q_values - target) ** 2
        per_critic_loss = jnp.mean(sq_err, axis=0)
        return jnp.sum(per_critic_loss), per_critic_loss

    (_, per_critic_loss), grads_tuple = jax.value_and_grad(
        all_critics_loss, has_aux=True,
    )(critic_params_tuple)

    new_critic_states = tuple(
        cs.apply_gradients(grads=g) for cs, g in zip(critic_states, grads_tuple)
    )
    training_state = training_state.replace(
        exploration_q_critic_states=new_critic_states,
    )
    return training_state, {"exploration_q_critic_loss": jnp.mean(per_critic_loss)}
