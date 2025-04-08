import jax
from jax import numpy as jnp
from netket.sampler.rules.base import MetropolisRule

class UniformRule(MetropolisRule):
    hilbert_space: jax.Array

    def __init__(
            self,
            hilbert_space: jnp.array,
    ):
        self.hilbert_space = hilbert_space

    def transition(rule, sampler, machine, params, sampler_state, key, σ):
        n_chains = σ.shape[0]
        key_arr = jax.random.split(key, n_chains)
        space_size = rule.hilbert_space.shape[0]

        @jax.jit
        def evol_state(σ, key):
            seq_j = jax.random.choice(key, space_size)
            σp = rule.hilbert_space[seq_j]
            return σp
        return (
            jax.vmap(evol_state, in_axes=(0,0), out_axes=0)(σ, key_arr),
            None,
         )

    def __repr__(self):
        return f"UniformRule()"