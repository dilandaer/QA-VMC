import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
PROPOSALMAT_PATH = os.path.join(PROJECT_ROOT, 'ProposalMat')
if PROPOSALMAT_PATH not in sys.path:
    sys.path.insert(0, PROPOSALMAT_PATH)
from ProposalMat import unpackSinglesDoubles_jax

import jax
from jax import numpy as jnp
from netket.sampler.rules.base import MetropolisRule

class ExcitationSDRule(MetropolisRule):
    sdexes_S: jax.Array
    sdexes_D: jax.Array
    # nonoAnoB: jax.Array
    noarr: jax.Array
    noAarr: jax.Array
    noBarr: jax.Array

    # length_sd: jax.Array

    def __init__(self, noarr, noAarr, noBarr):
        self.noarr = noarr
        self.noAarr = noAarr
        self.noBarr = noBarr
        no, noA, noB = noarr.shape[0], noAarr.shape[0], noBarr.shape[0]
        nvA = no - noA  # alpha 虚轨道数
        nvB = no - noB  # beta 虚轨道数
        nSa = noA * nvA  # 单激发的可能结果
        nSb = noB * nvB
        sdexes = unpackSinglesDoubles_jax(no, noA, noB)
        self.sdexes_S = sdexes[:nSa + nSb, :2]
        self.sdexes_D = sdexes[nSa + nSb:]
        # self.length_sd = jnp.sum(getNumSinglesDoubles(no, noA, noB))

    def transition(rule, sampler, machine, parameters, state, key, σ):
        n_chains = σ.shape[0]
        length_s, length_d = rule.sdexes_S.shape[0], rule.sdexes_D.shape[0]
        length_sd = length_s + length_d
        seq_arr = jax.random.choice(key, length_sd, shape=(n_chains,))
        no, noA, noB = rule.noarr.shape[0], rule.noAarr.shape[0], rule.noBarr.shape[0]

        def get_excitation_stateS(σ, seq):
            onv = jnp.array(σ * 0.5 + 0.5, dtype=jnp.bool_)
            length = onv.shape[0]
            onv_alpha = onv[::2]
            onv_beta = onv[1::2]
            idx_alpha_1 = jnp.nonzero(onv_alpha, size=noA)[0]
            idx_beta_1 = jnp.nonzero(onv_beta, size=noB)[0]
            idx_alpha_0 = jnp.nonzero(onv_alpha ^ True, size=no - noA)[0]
            idx_beta_0 = jnp.nonzero(onv_beta ^ True, size=no - noB)[0]
            idx_alpha = jnp.hstack((idx_alpha_1, idx_alpha_0))
            idx_beta = jnp.hstack((idx_beta_1, idx_beta_0))
            tmp = jnp.zeros_like(onv, dtype=jnp.int_)
            idx_r = tmp.at[jnp.hstack((jnp.arange(0, length, 2), jnp.arange(1, length, 2)))].set(
                jnp.hstack((2 * idx_alpha, 2 * idx_beta + 1)))
            σp = jnp.array(onv.at[idx_r[rule.sdexes_S[seq]]].set(jnp.array([0, 1], dtype=jnp.bool_)),
                           dtype=jnp.float_) * 2 - 1
            return jnp.array(σp, dtype=jnp.int8)

        def get_excitation_stateD(σ, seq):  # , noA, noB, no):
            onv = jnp.array(σ * 0.5 + 0.5, dtype=jnp.bool_)
            length = onv.shape[0]
            onv_alpha = onv[::2]
            onv_beta = onv[1::2]
            idx_alpha_1 = jnp.nonzero(onv_alpha, size=noA)[0]
            idx_beta_1 = jnp.nonzero(onv_beta, size=noB)[0]
            idx_alpha_0 = jnp.nonzero(onv_alpha ^ True, size=no - noA)[0]
            idx_beta_0 = jnp.nonzero(onv_beta ^ True, size=no - noB)[0]
            idx_alpha = jnp.hstack((idx_alpha_1, idx_alpha_0))
            idx_beta = jnp.hstack((idx_beta_1, idx_beta_0))
            tmp = jnp.zeros_like(onv, dtype=jnp.int_)
            idx_r = tmp.at[jnp.hstack((jnp.arange(0, length, 2), jnp.arange(1, length, 2)))].set(
                jnp.hstack((2 * idx_alpha, 2 * idx_beta + 1)))
            σp = jnp.array(onv.at[idx_r[rule.sdexes_D[seq - length_s]]].set(jnp.array([0, 1, 0, 1], dtype=jnp.bool_)),
                           dtype=jnp.float_) * 2 - 1
            return jnp.array(σp, dtype=jnp.int8)

        # get_excitation_state_obj = partial(get_excitation_state, noA=rule.noA, noB=rule.noB, no=rule.no)

        def decorator_case(σ, seq):
            return jax.lax.cond(seq < length_s, get_excitation_stateS, get_excitation_stateD, σ, seq)

        return (
            jax.vmap(decorator_case, in_axes=(0, 0), out_axes=0)(σ, seq_arr),
            None,
        )

    def __repr__(self):
        return f"ExcitationSDRule()"