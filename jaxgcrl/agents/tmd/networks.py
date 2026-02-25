import logging

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling


class TMDEncoder(nn.Module):
    """Encoder for TMD (Temporal Metric Distillation).

    Identical to the CRL Encoder but adds an optional ``value_exp``
    transformation at the output, which ensures non-negative representations
    required by the MRN / IQE quasimetric.
    """

    repr_dim: int = 512
    network_width: int = 512
    network_depth: int = 3
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = True
    value_exp: bool = True

    @nn.compact
    def __call__(self, data: jnp.ndarray) -> jnp.ndarray:
        logging.info("TMDEncoder input shape: %s", data.shape)
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        activation = nn.relu if self.use_relu else nn.swish

        x = data
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
            if self.use_ln:
                x = nn.LayerNorm()(x)
            x = activation(x)

            if self.skip_connections:
                if i == 0:
                    skip = x
                if i > 0 and i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        x = nn.Dense(self.repr_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        if self.value_exp:
            # Clip before exp for numerical stability (exp(-10)≈5e-5, exp(10)≈22026)
            x = jnp.exp(jnp.clip(x, -10.0, 10.0))

        return x
