import flax.linen as nn
import jax
import jax.numpy as jnp


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


def update_actor_and_alpha(config, networks, transitions, actor_state, critic_state, alpha_state, key):
    """Update actor and alpha. Accepts states directly for flexibility."""
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

        # Handle both single critic and ensemble cases
        is_ensemble = isinstance(critic_params["sa_encoder"], list)
        
        if is_ensemble:
            # For ensemble, compute mean Q-value across all critics
            all_qf_pi = []
            for i in range(len(critic_params["sa_encoder"])):
                sa_encoder_params = critic_params["sa_encoder"][i]
                g_encoder_params = critic_params["g_encoder"][i]
                sa_repr = networks["sa_encoder"].apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
                g_repr = networks["g_encoder"].apply(g_encoder_params, goal)
                qf_pi_i = energy_fn(config["energy_fn"], sa_repr, g_repr)
                all_qf_pi.append(qf_pi_i)
            qf_pi = jnp.mean(jnp.stack(all_qf_pi, axis=0), axis=0)
        else:
            sa_encoder_params, g_encoder_params = (
                critic_params["sa_encoder"],
                critic_params["g_encoder"],
            )
            sa_repr = networks["sa_encoder"].apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
            g_repr = networks["g_encoder"].apply(g_encoder_params, goal)
            qf_pi = energy_fn(config["energy_fn"], sa_repr, g_repr)

        per_sample_loss = jnp.exp(log_alpha) * log_prob - qf_pi

        actor_loss = jnp.mean(per_sample_loss)

        return actor_loss, log_prob

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - config["target_entropy"]))
        return jnp.mean(alpha_loss)
    
    batch_size = transitions.observation.shape[0]
    key, subkey = jax.random.split(key)
    sample_keys = jax.random.split(subkey, batch_size)

    (batch_actor_loss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
        actor_state.params,
        critic_state.params,
        alpha_state.params["log_alpha"],
        transitions,
        key,
    )

    # Update actor
    new_actor_state = actor_state.apply_gradients(grads=actor_grad)

    batch_alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss)(alpha_state.params, log_prob)
    new_alpha_state = alpha_state.apply_gradients(grads=alpha_grad)

    metrics = {
        "entropy": -log_prob,
        "actor_loss": batch_actor_loss,
        "alpha_loss": batch_alpha_loss,
        "log_alpha": new_alpha_state.params["log_alpha"],
    }

    # Only compute per-sample gradient statistics if adaptive mixing is enabled
    if config.get("use_adaptive_mixing", False):
        def single_sample_grad(i, key):
            single_transition = jax.tree_util.tree_map(lambda x: x[i], transitions)
            single_transition = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), single_transition)

            grad_fn = jax.grad(actor_loss, has_aux=True)
            return grad_fn(
                actor_state.params,
                critic_state.params,
                alpha_state.params["log_alpha"],
                single_transition,
                key
            )
        per_sample_grads, _ = jax.vmap(single_sample_grad)(jnp.arange(batch_size), sample_keys)

        was_proposed_mask = transitions.extras["state_extras"]["was_proposed_goal_mask"]

        def flatten_single_grad(i):
            single_grad = jax.tree_map(lambda x: x[i], per_sample_grads)
            flat, _ = jax.flatten_util.ravel_pytree(single_grad)
            return flat
        



    return new_actor_state, new_alpha_state, metrics


def update_critic(config, networks, transitions, critic_state, key):
    """Update critic(s). Supports both single critic and ensemble of critics. Accepts critic_state directly for flexibility."""
    
    critic_params = critic_state.params
    
    def single_critic_loss(sa_params, g_params, transitions, key):
        """Loss for a single critic."""
        state = transitions.observation[:, : config["state_size"]]
        action = transitions.action

        sa_repr = networks["sa_encoder"].apply(sa_params, jnp.concatenate([state, action], axis=-1))
        g_repr = networks["g_encoder"].apply(
            g_params, transitions.observation[:, config["state_size"] :]
        )

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

        return loss, (logsumexp, correct, logits_pos, logits_neg)
    
    # Check if we have an ensemble (list of params) or single critic
    is_ensemble = isinstance(critic_params["sa_encoder"], list)
    
    if is_ensemble:
        # Ensemble case: train each critic independently with the full optimizer
        num_ensemble = len(critic_params["sa_encoder"])
        
        def ensemble_loss(critic_params, transitions, key):
            """Combined loss for all ensemble members."""
            total_loss = 0.0
            all_logsumexp = []
            all_correct = []
            all_logits_pos = []
            all_logits_neg = []
            
            for i in range(num_ensemble):
                sa_params = critic_params["sa_encoder"][i]
                g_params = critic_params["g_encoder"][i]
                loss, (logsumexp, correct, logits_pos, logits_neg) = single_critic_loss(
                    sa_params, g_params, transitions, key
                )
                total_loss += loss
                all_logsumexp.append(logsumexp)
                all_correct.append(correct)
                all_logits_pos.append(logits_pos)
                all_logits_neg.append(logits_neg)
            
            avg_loss = total_loss / num_ensemble
            avg_logsumexp = jnp.mean(jnp.stack(all_logsumexp, axis=0), axis=0)
            avg_correct = jnp.mean(jnp.stack(all_correct, axis=0), axis=0)
            avg_logits_pos = jnp.mean(jnp.array(all_logits_pos))
            avg_logits_neg = jnp.mean(jnp.array(all_logits_neg))
            
            return avg_loss, (avg_logsumexp, avg_correct, avg_logits_pos, avg_logits_neg)
        
        (loss, (logsumexp, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
            ensemble_loss, has_aux=True
        )(critic_state.params, transitions, key)
        new_critic_state = critic_state.apply_gradients(grads=grad)
    else:
        # Single critic case (original implementation)
        def critic_loss(critic_params, transitions, key):
            return single_critic_loss(
                critic_params["sa_encoder"], 
                critic_params["g_encoder"], 
                transitions, 
                key
            )

        (loss, (logsumexp, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
            critic_loss, has_aux=True
        )(critic_state.params, transitions, key)
        new_critic_state = critic_state.apply_gradients(grads=grad)
    
    logsumexp_mean = logsumexp.mean()

    metrics = {
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp_mean,
        "critic_loss": loss,
    }

    return new_critic_state, metrics
