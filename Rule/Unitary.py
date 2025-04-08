import jax
from jax import numpy as jnp
from netket.sampler.rules.base import MetropolisRule

class UnitaryRule(MetropolisRule):
    Qmat: jax.Array
    hilbert_space: jax.Array
    two_vector: jax.Array
    idx: jax.Array

    def __init__(
            self,
            Qmat: jnp.array,
            hilbert_space: jnp.array,
    ):
        self.Qmat = Qmat
        nqubits = hilbert_space.shape[-1]
        self.two_vector = 2 ** jnp.arange(nqubits)[::-1]
        self.hilbert_space = hilbert_space
        self.idx = jnp.array(0.5 * hilbert_space + 0.5, dtype=jnp.int_)@self.two_vector

    def transition(rule, sampler, machine, params, sampler_state, key, σ):
        n_chains = σ.shape[0]
        key_arr = jax.random.split(key, n_chains)
        space_size = rule.hilbert_space.shape[0]

        @jax.jit
        def evol_state(σ, Q_mat, key):
            onv = jnp.array(0.5 * σ + 0.5, dtype=jnp.int_)
            index = onv @ rule.two_vector
            seq_i = jnp.where(rule.idx==index, size=1)[0][0]
            prob_arr = Q_mat[seq_i]
            seq_j = jax.random.choice(key, space_size, p = prob_arr)
            σp = rule.hilbert_space[seq_j]
            return σp
        #return (jnp.array(list(map(evol_state, σ, rule.Unitary_array[tuple([seq_arr])], statuses))), None,)
        return (
            jax.vmap(evol_state, in_axes=(0,None,0), out_axes=0)(σ, rule.Qmat, key_arr),
            None,
         )

    def __repr__(self):
        return f"UnitaryRule()"