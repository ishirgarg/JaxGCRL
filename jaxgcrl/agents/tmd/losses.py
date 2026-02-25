"""TMD (Temporal Metric Distillation) loss functions.

Implements:
  - MRN  (max-norm + L2 quasimetric)
  - IQE  (integrated quantile estimation quasimetric)
  - update_tmd_critic  – contrastive + backup (LINEX) + action-invariance losses
  - update_tmd_actor   – Q-maximisation + behavioural-cloning loss (no entropy)
"""

import jax
import jax.numpy as jnp
import optax


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def mrn_distance(x, y, components: int) -> jnp.ndarray:
    """MRN quasimetric distance.

    Supports arbitrary leading batch dimensions and broadcasting.
    The last axis is the representation dimension (must be divisible by
    ``components``).

    Returns an array with the last axis removed.
    """
    K = components

    def mrn_component(x_k, y_k):
        eps = 1e-6
        d = x_k.shape[-1]
        mask = jnp.arange(d) < d // 2
        max_part = jnp.max(jax.nn.relu((x_k - y_k) * mask), axis=-1)
        l2_part = jnp.sqrt(jnp.square((x_k - y_k) * (1 - mask)).sum(axis=-1) + eps)
        return max_part + l2_part

    # (..., D/K, K)  – stack splits along a new trailing axis
    x_split = jnp.stack(jnp.split(x, K, axis=-1), axis=-1)
    y_split = jnp.stack(jnp.split(y, K, axis=-1), axis=-1)
    # vmap over each component (last axis)  → (..., K)
    dists = jax.vmap(mrn_component, in_axes=(-1, -1), out_axes=-1)(x_split, y_split)
    return dists.mean(axis=-1)


def iqe_distance(x, y, k: int, alpha_raw: jnp.ndarray) -> jnp.ndarray:
    """IQE (integrated quantile estimation) quasimetric distance.

    ``alpha_raw`` is a *trainable* scalar (un-sigmoided).
    Supports arbitrary leading dimensions via broadcasting.
    """
    alpha = jax.nn.sigmoid(alpha_raw)
    D_orig = x.shape[-1]
    # Reshape last dim: D  →  (D//k, k)
    reshape_tail = (D_orig // k, k)
    x_r = jnp.reshape(x, (*x.shape[:-1], *reshape_tail))
    y_r = jnp.reshape(y, (*y.shape[:-1], *reshape_tail))
    valid = x_r < y_r            # (..., D//k, k)  bool
    D = x_r.shape[-1]            # = k  (after reshape)

    # Concatenate along last axis: (..., D//k, 2k)
    xy = jnp.concatenate(jnp.broadcast_arrays(x_r, y_r), axis=-1)
    ixy = xy.argsort(axis=-1)
    sxy = jnp.take_along_axis(xy, ixy, axis=-1)
    neg_inc_copies = (
        jnp.take_along_axis(valid, ixy % D, axis=-1)
        * jnp.where(ixy < D, -1, 1)
    )
    neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
    neg_f = (neg_inp_copies < 0) * (-1.0)
    neg_incf = jnp.concatenate(
        [neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1
    )
    components = (sxy * neg_incf).sum(-1)       # (..., D//k)
    result = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)
    return result                                # (...)


def tmd_distance(x, y, components: int, use_iqe: bool,
                 iqe_alpha_raw=None) -> jnp.ndarray:
    """Dispatch to MRN or IQE depending on ``use_iqe``."""
    if use_iqe:
        return iqe_distance(x, y, components, iqe_alpha_raw)
    else:
        return mrn_distance(x, y, components)


# ---------------------------------------------------------------------------
# Critic update
# ---------------------------------------------------------------------------

def update_tmd_critic(config, networks, transitions, critic_state, key):
    """Compute TMD critic loss and apply one gradient step.

    The critic loss is a weighted sum of three terms (matching OGBench TMD):
      1. Contrastive (InfoNCE) loss between phi(s,a) and psi(g)
      2. Action-invariance loss: d(psi(s_goal), phi(s,a))
      3. Backup (LINEX) loss: temporal contraction of the quasimetric

    Critic params must have keys: ``phi_1``, ``phi_2``, ``psi_1``, ``psi_2``,
    and optionally ``iqe_alpha_raw`` when ``use_iqe=True``.
    """
    state_size = config["state_size"]
    goal_indices = config["goal_indices"]
    K = config["tmd_components"]
    use_iqe = config["use_iqe"]

    def critic_loss_fn(critic_params, transitions):
        state = transitions.observation[:, :state_size]               # (B, S)
        action = transitions.action                                    # (B, A)
        goal = transitions.observation[:, state_size:]                 # (B, G)
        # Immediate next state projected to goal space
        next_state = transitions.extras["next_state"]                  # (B, S)
        next_state_goal = next_state[:, goal_indices]                  # (B, G)
        # Current state projected to goal space (for invariance loss)
        state_goal = state[:, goal_indices]                            # (B, G)

        batch_size = state.shape[0]
        D = config["latent_dim"]

        sa = jnp.concatenate([state, action], axis=-1)                # (B, S+A)

        # ---- Ensemble representations (2, B, D) --------------------------
        phi_1 = networks["phi"].apply(critic_params["phi_1"], sa)
        phi_2 = networks["phi"].apply(critic_params["phi_2"], sa)
        phi = jnp.stack([phi_1, phi_2], axis=0)                       # (2, B, D)

        psi_s_1 = networks["psi"].apply(critic_params["psi_1"], state_goal)
        psi_s_2 = networks["psi"].apply(critic_params["psi_2"], state_goal)
        psi_s = jnp.stack([psi_s_1, psi_s_2], axis=0)                 # (2, B, D)

        psi_next_1 = networks["psi"].apply(critic_params["psi_1"], next_state_goal)
        psi_next_2 = networks["psi"].apply(critic_params["psi_2"], next_state_goal)
        psi_next = jnp.stack([psi_next_1, psi_next_2], axis=0)        # (2, B, D)

        psi_g_1 = networks["psi"].apply(critic_params["psi_1"], goal)
        psi_g_2 = networks["psi"].apply(critic_params["psi_2"], goal)
        psi_g = jnp.stack([psi_g_1, psi_g_2], axis=0)                 # (2, B, D)

        iqe_alpha_raw = critic_params.get("iqe_alpha_raw", None) if use_iqe else None

        def dist(x, y):
            return tmd_distance(x, y, K, use_iqe, iqe_alpha_raw)

        # ---- 1. Contrastive (InfoNCE) loss --------------------------------
        # pairwise: phi[:, :, None, :] (2, B, 1, D) x psi_g[:, None, :, :] (2, 1, B, D)
        dist_mat = dist(phi[:, :, None, :], psi_g[:, None, :, :])     # (2, B, B)
        logits = -dist_mat / jnp.sqrt(D)

        I = jnp.eye(batch_size)
        contrastive_loss = jnp.mean(
            jax.vmap(
                lambda lg: optax.softmax_cross_entropy(logits=lg.T, labels=I)
            )(logits)
        )

        # ---- 2. Action-invariance loss ------------------------------------
        phi_inv = jax.lax.stop_gradient(phi) if config["stopgrad_phi_invariance"] else phi
        inv_dist = dist(psi_s, phi_inv)                                # (2, B)
        invariance_loss = jnp.mean(inv_dist)

        # ---- 3. Backup (LINEX) loss ---------------------------------------
        dist_next = dist(psi_next[:, :, None, :], psi_g[:, None, :, :])   # (2, B, B)
        psi_g_bk = jax.lax.stop_gradient(psi_g) if config["stopgrad_psi_backup"] else psi_g
        dist_backup = dist(phi[:, :, None, :], psi_g_bk[:, None, :, :])   # (2, B, B)
        dist_next_sg = jax.lax.stop_gradient(dist_next)

        t = config["t"]
        gamma = config["discounting"]
        delta = dist_backup - dist_next_sg
        mask = delta > t
        delta_clipped = jnp.where(mask, t, delta)
        divergence = jnp.where(mask, delta, gamma * jnp.exp(delta_clipped) - dist_backup)

        dw = config["diag_backup"]
        # Mix off-diagonal (random pairs) with diagonal (same-trajectory pairs)
        divergence = (
            divergence * (1 - dw)
            + jnp.diagonal(divergence, axis1=1, axis2=2)[..., None] * dw
        )
        backup_loss = jnp.mean(divergence)

        # ---- Total loss --------------------------------------------------
        total_loss = (
            contrastive_loss
            + config["zeta"] * invariance_loss
            + config["zeta"] * backup_loss
        )

        # ---- Metrics -----------------------------------------------------
        logits_mean = jnp.mean(logits, axis=0)                        # (B, B)
        correct = jnp.argmax(logits_mean, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits_mean * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits_mean * (1 - I)) / jnp.sum(1 - I)

        return total_loss, (
            contrastive_loss, backup_loss, invariance_loss,
            correct, logits_pos, logits_neg, dist_mat.mean(),
        )

    (loss, (contrastive_loss, backup_loss, invariance_loss,
            correct, logits_pos, logits_neg, dist_mean)), grad = (
        jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_state.params, transitions)
    )
    new_critic_state = critic_state.apply_gradients(grads=grad)

    metrics = {
        "critic_loss": loss,
        "contrastive_loss": contrastive_loss,
        "backup_loss": backup_loss,
        "invariance_loss": invariance_loss,
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "dist_mean": dist_mean,
    }
    return new_critic_state, metrics


# ---------------------------------------------------------------------------
# Actor update
# ---------------------------------------------------------------------------

def update_tmd_actor(config, networks, transitions, actor_state, critic_state, key):
    """Compute TMD actor loss and apply one gradient step.

    Loss = Q-maximisation (normalised) + alpha * BC.
    No entropy term – alpha is a fixed BC coefficient (not an adaptive entropy
    coefficient as in SAC).
    """
    state_size = config["state_size"]
    K = config["tmd_components"]
    use_iqe = config["use_iqe"]

    def actor_loss_fn(actor_params, critic_params, transitions, key):
        state = transitions.observation[:, :state_size]
        goal = transitions.observation[:, state_size:]
        obs = jnp.concatenate([state, goal], axis=-1)

        means, log_stds = networks["actor"].apply(actor_params, obs)
        stds = jnp.exp(log_stds)

        # Q-actions: use mode for const_std, otherwise sample
        if config["const_std"]:
            q_actions = jnp.clip(means, -1, 1)
        else:
            q_actions = jnp.clip(
                means + stds * jax.random.normal(key, means.shape), -1, 1
            )

        sa = jnp.concatenate([state, q_actions], axis=-1)
        iqe_alpha_raw = critic_params.get("iqe_alpha_raw", None) if use_iqe else None

        phi_1 = networks["phi"].apply(critic_params["phi_1"], sa)
        phi_2 = networks["phi"].apply(critic_params["phi_2"], sa)
        psi_g_1 = networks["psi"].apply(critic_params["psi_1"], goal)
        psi_g_2 = networks["psi"].apply(critic_params["psi_2"], goal)

        q1 = -tmd_distance(phi_1, psi_g_1, K, use_iqe, iqe_alpha_raw)   # (B,)
        q2 = -tmd_distance(phi_2, psi_g_2, K, use_iqe, iqe_alpha_raw)   # (B,)
        q = jnp.minimum(q1, q2)

        # Scale-invariant Q loss
        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)

        # BC loss: log-probability of stored (tanh-squashed) actions
        stored_actions = transitions.action
        pre_tanh = jnp.arctanh(jnp.clip(stored_actions, -0.999, 0.999))
        log_prob = jax.scipy.stats.norm.logpdf(pre_tanh, loc=means, scale=stds)
        log_prob -= jnp.log(1 - jnp.square(stored_actions) + 1e-6)
        log_prob = log_prob.sum(-1)                                    # (B,)
        bc_loss = -(config["alpha"] * log_prob).mean()

        actor_loss = q_loss + bc_loss
        return actor_loss, (q.mean(), jnp.abs(q).mean(), log_prob.mean())

    (loss, (q_mean, q_abs_mean, log_prob_mean)), grad = jax.value_and_grad(
        actor_loss_fn, has_aux=True
    )(actor_state.params, critic_state.params, transitions, key)

    new_actor_state = actor_state.apply_gradients(grads=grad)

    metrics = {
        "actor_loss": loss,
        "q_mean": q_mean,
        "q_abs_mean": q_abs_mean,
        "bc_log_prob": log_prob_mean,
    }
    return new_actor_state, metrics
