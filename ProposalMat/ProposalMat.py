import jax
import jax.numpy as jnp
import itertools
import numpy as np
import scipy
import time

jax.config.update('jax_enable_x64',True)


def unpack_canon(ij_arr):
    """
    Unpack a packed index array into two separate indices (i and j) such that i > j.

    Parameters:
    -----------
    ij_arr : array_like
        A 1D array of packed indices.

    Returns:
    --------
    i_arr, j_arr : ndarray
        Two arrays containing the unpacked indices corresponding to i and j.
    """
    # Compute i from the packed index (the formula ensures i > j)
    i_arr = jnp.int_(jnp.sqrt((ij_arr + 1) * 2) + 0.5)
    # Compute j using the relation for packed indices
    j_arr = ij_arr - i_arr * (i_arr - 1) // 2
    return i_arr, j_arr


def unpackSinglesDoubles_jax(k, noA, noB):
    """
    Generate indices for all possible single and double excitations based on a given configuration.

    This function computes the excitation indices for:
      - Single excitations (separated for alpha and beta spins)
      - Double excitations, including:
            - Alpha-alpha double excitations
            - Beta-beta double excitations
            - Mixed alpha-beta double excitations

    The indices follow a specific convention:
      - Even numbers correspond to alpha spin orbitals.
      - Odd numbers correspond to beta spin orbitals.
      - For single excitations, unused indices are set to -1.

    Parameters:
    -----------
    k : int
        Total number of spin orbitals.
    noA : int
        Number of occupied alpha orbitals.
    noB : int
        Number of occupied beta orbitals.

    Returns:
    --------
    excitations : ndarray, shape (N, 4)
        An array of excitation indices. Each row represents an excitation in the form [i, a, j, b],
        where:
          - For single excitations: j and b are set to -1.
          - For double excitations: i, j are occupied orbital indices, and a, b are virtual orbital indices.

    Notes:
    -----
    - The number of alpha virtual orbitals is calculated as nvA = k - noA,
      and similarly for beta, nvB = k - noB.
    - The excitations are grouped into five cases:
         case0: alpha single excitations,
         case1: beta single excitations,
         case2: alpha-alpha double excitations,
         case3: beta-beta double excitations,
         case4: mixed alpha-beta double excitations.
    - The function uses JAX's vectorized mapping (vmap) and lax.switch for efficient computation.
    """

    # Calculate the number of virtual orbitals for alpha and beta spins.
    nvA = k - noA  # number of alpha virtual orbitals
    nvB = k - noB  # number of beta virtual orbitals

    # Calculate the number of possible single excitations.
    nSa = noA * nvA  # alpha single excitations
    nSb = noB * nvB  # beta single excitations

    # Calculate the number of possible double excitations.
    noAA = noA * (noA - 1) // 2  # number of ways to choose two occupied alpha orbitals
    nvAA = nvA * (nvA - 1) // 2  # number of ways to choose two virtual alpha orbitals
    noBB = noB * (noB - 1) // 2  # number of ways to choose two occupied beta orbitals
    nvBB = nvB * (nvB - 1) // 2  # number of ways to choose two virtual beta orbitals
    novAA = noA * nvA  # intermediate for mixed excitations
    novBB = noB * nvB
    nDaa = noAA * nvAA  # alpha-alpha double excitations
    nDbb = noBB * nvBB  # beta-beta double excitations
    nDab = novAA * novBB  # alpha-beta double excitations

    # Pack the dimensions for each excitation type into an array.
    dims = jnp.array([nSa, nSb, nDaa, nDbb, nDab])
    # Create an array of all indices (from 0 to total number of excitations - 1)
    idxes = jnp.arange(jnp.sum(dims))
    # Compute cumulative sums to determine boundaries between cases.
    d_arr = jnp.cumsum(dims)

    # Determine which excitation case each index belongs to.
    i0 = jnp.int_(idxes >= d_arr[0])
    i1 = jnp.int_(idxes >= d_arr[1])
    i2 = jnp.int_(idxes >= d_arr[2])
    i3 = jnp.int_(idxes >= d_arr[3])
    # The case index will be 0 for alpha singles, 1 for beta singles, etc.
    case_arr = i0 + i1 + i2 + i3

    # Define functions for each case, returning [i, a, j, b] for each excitation.

    def case0(idx):
        # Alpha single excitation:
        # i: occupied alpha orbital, a: virtual alpha orbital
        i = 2 * (idx % noA)
        a = 2 * (idx // noA + noA)
        # For single excitations, j and b are unused (set to -1)
        j = -1
        b = -1
        return jnp.array([i, a, j, b], dtype=jnp.int_)

    def case1(idx):
        # Beta single excitation:
        jdx = idx - d_arr[0]
        i = 2 * (jdx % noB) + 1  # occupied beta orbitals (odd index)
        a = 2 * (jdx // noB + noB) + 1  # virtual beta orbitals (odd index)
        j = -1
        b = -1
        return jnp.array([i, a, j, b], dtype=jnp.int_)

    def case2(idx):
        # Alpha-alpha double excitations:
        jdx = idx - d_arr[1]
        ijA = jdx % noAA  # packed index for two occupied alpha orbitals
        abA = jdx // noAA  # packed index for two virtual alpha orbitals
        i, j = unpack_canon(ijA)  # Unpack occupied indices (i > j)
        a, b = unpack_canon(abA)  # Unpack virtual indices (a > b)
        i = 2 * i  # even indices for alpha
        j = 2 * j
        a = 2 * (a + noA)  # offset for virtual orbitals
        b = 2 * (b + noA)
        return jnp.array([i, a, j, b], dtype=jnp.int_)

    def case3(idx):
        # Beta-beta double excitations:
        jdx = idx - d_arr[2]
        ijB = jdx % noBB  # packed index for two occupied beta orbitals
        abB = jdx // noBB  # packed index for two virtual beta orbitals
        i, j = unpack_canon(ijB)  # Unpack occupied indices (i > j)
        a, b = unpack_canon(abB)  # Unpack virtual indices (a > b)
        i = 2 * i + 1  # odd indices for beta
        j = 2 * j + 1
        a = 2 * (a + noB) + 1
        b = 2 * (b + noB) + 1
        return jnp.array([i, a, j, b], dtype=jnp.int_)

    def case4(idx):
        # Mixed alpha-beta double excitations:
        jdx = idx - d_arr[3]
        iaA = jdx % (noA * nvA)  # index for alpha single excitation component
        jbB = jdx // (noA * nvA)  # index for beta single excitation component
        i = 2 * (iaA % noA)
        a = 2 * (iaA // noA + noA)
        j = 2 * (jbB % noB) + 1
        b = 2 * (jbB // noB + noB) + 1
        return jnp.array([i, a, j, b], dtype=jnp.int_)

    # Decorator function to select the proper case based on the computed case index.
    def decorator_case(indice, idx):
        return jax.lax.switch(indice, [case0, case1, case2, case3, case4], idx)

    # Use vectorized mapping (vmap) to apply the decorator function to all indices.
    return jax.vmap(decorator_case, in_axes=(0, 0))(case_arr, idxes)

def getNumSinglesDoubles(k, noA, noB):
    """
    Compute the number of possible single and double excitations for a given configuration.

    The function calculates the following excitation counts:
      - nSa: Number of alpha single excitations (occupied alpha -> virtual alpha)
      - nSb: Number of beta single excitations (occupied beta -> virtual beta)
      - nDaa: Number of alpha-alpha double excitations (two alpha electrons excited)
      - nDbb: Number of beta-beta double excitations (two beta electrons excited)
      - nDab: Number of mixed alpha-beta double excitations

    Parameters:
    -----------
    k : int
        Total number of spin orbitals.
    noA : int
        Number of occupied alpha orbitals.
    noB : int
        Number of occupied beta orbitals.

    Returns:
    --------
    jnp.array
        A JAX array containing the counts [nSa, nSb, nDaa, nDbb, nDab] for each excitation type.
    """
    # Number of virtual orbitals for alpha and beta spins
    nvA = k - noA  # alpha virtual orbitals
    nvB = k - noB  # beta virtual orbitals

    # Count single excitations: occupied -> virtual transitions for each spin
    nSa = noA * nvA  # alpha singles
    nSb = noB * nvB  # beta singles

    # Count double excitations:
    # For alpha-alpha: Choose 2 occupied out of noA and 2 virtual out of nvA
    nDaa = noA * (noA - 1) // 2 * nvA * (nvA - 1) // 2
    # For beta-beta: Choose 2 occupied out of noB and 2 virtual out of nvB
    nDbb = noB * (noB - 1) // 2 * nvB * (nvB - 1) // 2
    # For mixed alpha-beta: Every occupied alpha can be paired with every occupied beta,
    # and similarly for virtual orbitals.
    nDab = noA * noB * nvA * nvB

    return jnp.array([nSa, nSb, nDaa, nDbb, nDab])

def getNumSinglesDoubles(k, noA, noB):
    """
    Compute the number of possible single and double excitations for a given configuration.

    The function calculates the following excitation counts:
      - nSa: Number of alpha single excitations (occupied alpha -> virtual alpha)
      - nSb: Number of beta single excitations (occupied beta -> virtual beta)
      - nDaa: Number of alpha-alpha double excitations (two alpha electrons excited)
      - nDbb: Number of beta-beta double excitations (two beta electrons excited)
      - nDab: Number of mixed alpha-beta double excitations

    Parameters:
    -----------
    k : int
        Total number of spin orbitals.
    noA : int
        Number of occupied alpha orbitals.
    noB : int
        Number of occupied beta orbitals.

    Returns:
    --------
    jnp.array
        A JAX array containing the counts [nSa, nSb, nDaa, nDbb, nDab] for each excitation type.
    """
    # Number of virtual orbitals for alpha and beta spins
    nvA = k - noA  # alpha virtual orbitals
    nvB = k - noB  # beta virtual orbitals

    # Count single excitations: occupied -> virtual transitions for each spin
    nSa = noA * nvA  # alpha singles
    nSb = noB * nvB  # beta singles

    # Count double excitations:
    # For alpha-alpha: Choose 2 occupied out of noA and 2 virtual out of nvA
    nDaa = noA * (noA - 1) // 2 * nvA * (nvA - 1) // 2
    # For beta-beta: Choose 2 occupied out of noB and 2 virtual out of nvB
    nDbb = noB * (noB - 1) // 2 * nvB * (nvB - 1) // 2
    # For mixed alpha-beta: Every occupied alpha can be paired with every occupied beta,
    # and similarly for virtual orbitals.
    nDab = noA * noB * nvA * nvB

    return jnp.array([nSa, nSb, nDaa, nDbb, nDab])


def get_excitation_stateS(onv, sdexes_Sseq, no, noA, noB):
    """
    Generate a new occupation number vector (ONV) corresponding to a single excitation.

    This function takes an input ONV (as a boolean array) and modifies it based on the
    provided single-excitation indices. The occupation vector is assumed to be arranged in an
    interleaved manner for alpha and beta spins.

    Parameters:
    -----------
    onv : array_like
        The original occupation number vector (ONV) as a boolean array.
    sdexes_Sseq : array_like
        A 2-element sequence indicating the positions (indices) in the re-ordered ONV that
        are to be modified for the single excitation.
    no : int
        Total number of spatial orbitals.
    noA : int
        Number of occupied alpha orbitals.
    noB : int
        Number of occupied beta orbitals.

    Returns:
    --------
    onv_new : jnp.array
        The updated ONV (as an integer array) after applying the single excitation.
        For single excitations, the unused indices are left unchanged.
    """
    # Convert the original ONV to a boolean JAX array.
    onv = jnp.array(onv, dtype=jnp.bool_)
    length = onv.shape[0]

    # Split ONV into alpha and beta components (assuming interleaved ordering: alpha at even positions, beta at odd)
    onv_alpha = onv[::2]
    onv_beta = onv[1::2]

    # Get indices where orbitals are occupied (True) for both spins.
    idx_alpha_1 = jnp.nonzero(onv_alpha, size=noA)[0]
    idx_beta_1 = jnp.nonzero(onv_beta, size=noB)[0]

    # Get indices where orbitals are unoccupied (False) for both spins.
    idx_alpha_0 = jnp.nonzero(~onv_alpha, size=no - noA)[0]
    idx_beta_0 = jnp.nonzero(~onv_beta, size=no - noB)[0]

    # Concatenate occupied and unoccupied indices for reordering.
    idx_alpha = jnp.hstack((idx_alpha_1, idx_alpha_0))
    idx_beta = jnp.hstack((idx_beta_1, idx_beta_0))

    # Create a temporary array and re-map the indices for both spins.
    tmp = jnp.zeros_like(onv, dtype=jnp.int_)
    # For alpha: positions 0, 2, 4,...; for beta: positions 1, 3, 5,...
    idx_r = tmp.at[jnp.hstack((jnp.arange(0, length, 2), jnp.arange(1, length, 2)))] \
                .set(jnp.hstack((2 * idx_alpha, 2 * idx_beta + 1)))

    # Modify the ONV at positions specified by sdexes_Sseq.
    # Here, setting these positions to [0, 1] indicates the excitation process.
    onv_new = jnp.array(onv.at[idx_r[sdexes_Sseq]].set(jnp.array([0, 1], dtype=jnp.bool_)),
                        dtype=jnp.int_)
    return onv_new


def get_excitation_stateD(onv, sdexes_Dseq, no, noA, noB):
    """
    Generate a new occupation number vector (ONV) corresponding to a double excitation.

    Similar to `get_excitation_stateS`, this function updates the ONV based on the provided
    double-excitation indices. The ONV is expected to be arranged in an interleaved manner for alpha and beta spins.

    Parameters:
    -----------
    onv : array_like
        The original occupation number vector (ONV) as a boolean array.
    sdexes_Dseq : array_like
        A sequence indicating the positions (indices) in the re-ordered ONV that
        are to be modified for the double excitation.
    no : int
        Total number of spatial orbitals.
    noA : int
        Number of occupied alpha orbitals.
    noB : int
        Number of occupied beta orbitals.

    Returns:
    --------
    onv_new : jnp.array
        The updated ONV (as an integer array) after applying the double excitation.
    """
    # Convert the original ONV to a boolean JAX array.
    onv = jnp.array(onv, dtype=jnp.bool_)
    length = onv.shape[0]

    # Separate the ONV into alpha and beta parts (even indices for alpha, odd for beta).
    onv_alpha = onv[::2]
    onv_beta = onv[1::2]

    # Obtain indices of occupied and unoccupied orbitals for both spins.
    idx_alpha_1 = jnp.nonzero(onv_alpha, size=noA)[0]
    idx_beta_1 = jnp.nonzero(onv_beta, size=noB)[0]
    idx_alpha_0 = jnp.nonzero(~onv_alpha, size=no - noA)[0]
    idx_beta_0 = jnp.nonzero(~onv_beta, size=no - noB)[0]

    # Reorder the indices by concatenating the occupied and unoccupied parts.
    idx_alpha = jnp.hstack((idx_alpha_1, idx_alpha_0))
    idx_beta = jnp.hstack((idx_beta_1, idx_beta_0))

    # Create a temporary index array for remapping.
    tmp = jnp.zeros_like(onv, dtype=jnp.int_)
    idx_r = tmp.at[jnp.hstack((jnp.arange(0, length, 2), jnp.arange(1, length, 2)))] \
                .set(jnp.hstack((2 * idx_alpha, 2 * idx_beta + 1)))

    # For double excitations, update the ONV at specified positions to [0, 1, 0, 1]
    # representing the two-electron excitation process.
    onv_new = jnp.array(onv.at[idx_r[sdexes_Dseq]].set(jnp.array([0, 1, 0, 1], dtype=jnp.bool_)),
                        dtype=jnp.int_)
    return onv_new


def get_connect_ExactationSD(onv, no, noA, noB, sdexes):
    """
    Compute the connected excitation states (both single and double excitations) and return their integer indices.

    This function combines single and double excitation modifications of the given occupation
    number vector (ONV) based on provided excitation sequences. It uses JAX's vectorized mapping to
    generate the new ONVs, then converts each resulting ONV (represented as a binary vector) to a unique
    integer index.

    Parameters:
    -----------
    onv : array_like
        The original occupation number vector (ONV) as a boolean array.
    no : int
        Total number of spatial orbitals.
    noA : int
        Number of occupied alpha orbitals.
    noB : int
        Number of occupied beta orbitals.
    sdexes : array_like
        A combined sequence of excitation indices for single and double excitations.
        The first portion corresponds to singles (both alpha and beta), and the remaining indices correspond to doubles.

    Returns:
    --------
    idx : jnp.array
        A JAX array of integer indices, each uniquely representing an excited ONV.
        The conversion to an integer is performed via a weighted sum over binary digits (with 2 raised to a power).
    """
    # Calculate the number of virtual orbitals for alpha and beta spins.
    nvA = no - noA  # alpha virtual orbitals
    nvB = no - noB  # beta virtual orbitals

    # Compute the number of single excitations for alpha and beta.
    nSa = noA * nvA  # alpha singles
    nSb = noB * nvB  # beta singles

    # Split the provided excitation indices into singles and doubles.
    sdexes_S = sdexes[:nSa + nSb, :2]  # first part for singles (2 indices per excitation)
    sdexes_D = sdexes[nSa + nSb:]       # remainder for doubles

    # Apply the excitation functions in a vectorized fashion for singles and doubles.
    onv_new_S = jax.vmap(get_excitation_stateS, in_axes=(None, 0, None, None, None))(onv, sdexes_S, no, noA, noB)
    onv_new_D = jax.vmap(get_excitation_stateD, in_axes=(None, 0, None, None, None))(onv, sdexes_D, no, noA, noB)

    # Combine the new ONVs for singles and doubles.
    onv_new = jnp.vstack((onv_new_S, onv_new_D))

    # Calculate the total number of qubits (each spatial orbital has 2 spin orbitals).
    nqubits = 2 * no

    # Create a weight vector for binary-to-integer conversion.
    # Each bit is assigned a weight 2^(position), and we reverse the order.
    two_vectors = (2 ** np.arange(nqubits))[::-1]

    # Compute the integer index corresponding to each ONV by taking a dot product with the weight vector.
    idx = onv_new @ two_vectors

    return idx

########################################### ExcitationSD Proposal ##################################################

def get_all_ExactationSD_index(idx_conv, no, noA, noB):
    """
    Compute the indices of all connected excitation states (both singles and doubles) for a given occupation configuration.

    This function converts an integer representation of an occupation number vector (ONV) into its binary form,
    applies a set of precomputed single and double excitation mappings (generated by `unpackSinglesDoubles_jax`),
    and returns the unique integer indices corresponding to each excited ONV.

    Parameters:
    ----------
    idx_conv : array_like
        An integer or an array of integers representing the occupation number vector(s) (ONVs) in their
        compact integer form.
    no : int
        Total number of spatial orbitals.
    noA : int
        Number of occupied alpha orbitals.
    noB : int
        Number of occupied beta orbitals.

    Returns:
    -------
    jnp.array
        An array of integer indices, each uniquely representing an excited ONV after applying
        both single and double excitations.
    """
    # Generate the sequence of excitation indices for singles and doubles.
    sdexes = unpackSinglesDoubles_jax(no, noA, noB)
    # Convert the integer representation of the ONV into its binary form.
    onv_new = jnp.mod(jnp.right_shift(idx_conv[..., None], jnp.arange(2 * no)[::-1]), 2)
    # Compute the connected excitation indices using vectorized mapping of get_connect_ExactationSD.
    return jax.vmap(get_connect_ExactationSD, in_axes=(0, None, None, None, None), out_axes=0)(onv_new, no, noA, noB,
                                                                                               sdexes)


def get_Q_excitationSD_mat(idx_conv, no, noA, noB):
    """
    Compute the proposal matrix Q for MCMC based on single and double excitations.

    This function constructs a sparse proposal matrix Q that describes the probability
    of transitioning from one occupation number vector (ONV) configuration to another via
    excitations (both single and double). The excitations are randomly chosen among all
    possible connected excitations, and the proposal probability for each connected state
    is uniformly weighted.

    Parameters:
    -----------
    idx_conv : array_like
        An array of integers representing the current occupation number vector (ONV) configurations
        in their compact integer form.
    no : int
        Total number of spatial orbitals.
    noA : int
        Number of occupied alpha orbitals.
    noB : int
        Number of occupied beta orbitals.

    Returns:
    -------
    np.array
        A dense numpy array representing the proposal matrix Q for the given configurations,
        where Q[i, j] is the proposal probability of transitioning from configuration i to configuration j.
    """
    # Get the matrix of connected excitation indices for all configurations
    loca_mat = get_all_ExactationSD_index(idx_conv, no, noA, noB)
    # The number of possible excitations per configuration
    repeat_length = loca_mat.shape[1]

    # Create a repeated array for current configuration indices to match the shape of loca_mat.
    idx_conv_trans_arr = np.repeat(idx_conv.reshape(-1, 1), repeats=repeat_length, axis=1).flatten()
    # Flatten the connected excitation indices matrix.
    loca_arr = loca_mat.flatten()

    # Each possible excitation is equally likely; compute the probability weight.
    repeat_length_arr = np.ones_like(loca_arr) * (1 / repeat_length)

    # Build the sparse proposal matrix Q using the (row, col, data) format.
    Q_mat_tot = scipy.sparse.csr_matrix((repeat_length_arr, (idx_conv_trans_arr, loca_arr)))

    # Extract and return the dense submatrix corresponding to the given configurations.
    return np.array(Q_mat_tot[idx_conv][:, idx_conv].todense())

########################################## ExcitationSD+flip Proposal ##############################################
def get_connect_flip_index(k_bin_arr, nqubits):
    """
    Compute the bit-flip connected configuration index.

    This function flips all bits in the input binary configuration and converts it
    back to its integer representation.

    Parameters
    ----------
    k_bin_arr : ndarray of shape (nqubits,)
        Binary occupation vector representing a quantum configuration.

    nqubits : int
        Total number of qubits (length of the occupation vector).

    Returns
    -------
    index : int
        Integer representation of the flipped bitstring.
    """
    flip_arr = k_bin_arr ^ 1
    two_vector = (2 ** np.arange(nqubits))[::-1]
    index = flip_arr @ two_vector
    return index


def get_all_flip_index(idx_conv, nqubits):
    """
    Compute bit-flip indices for a batch of occupation states.

    Parameters
    ----------
    idx_conv : ndarray of shape (n_configs,)
        Integer-encoded occupation numbers.

    nqubits : int
        Number of qubits in each configuration.

    Returns
    -------
    flipped_indices : ndarray of shape (n_configs,)
        Integer indices of configurations after applying global bit-flip.
    """
    onv_new = jnp.mod(jnp.right_shift(idx_conv[..., None], jnp.arange(nqubits)[::-1]), 2)
    return jax.vmap(get_connect_flip_index, in_axes=(0, None), out_axes=0)(onv_new, nqubits)


def get_Q_excitationSD_flip_mat(idx_conv, no, noA, noB):
    """
    Construct the transition matrix Q for MCMC proposals including both excitation and flip moves.

    This function combines all single/double excitation transitions and global bit-flip transitions
    into a sparse transition matrix Q, normalized appropriately so each row sums to 1.

    Parameters
    ----------
    idx_conv : ndarray of shape (n_states,)
        List of state indices (basis states) over which to build the Q matrix.

    no : int
        Number of spatial orbitals (half of total qubits).

    noA : int
        Number of occupied alpha-spin orbitals.

    noB : int
        Number of occupied beta-spin orbitals.

    Returns
    -------
    Q_mat : ndarray of shape (n_states, n_states)
        Dense transition probability matrix, combining excitation and bit-flip moves.
        Each row is normalized to sum to 1.

    Notes
    -----
    - Excitations are generated using all possible valid single and double substitutions.
    - Global spin-flip move is added with equal probability to maintain ergodicity.
    - Resulting matrix is symmetric if the excitation and flip proposals are symmetrically defined.
    """
    loca_mat = get_all_ExactationSD_index(idx_conv, no, noA, noB)
    loca_mat_flip = get_all_flip_index(idx_conv, 2 * no)
    repeat_length = loca_mat.shape[1]
    idx_conv_trans_arr = np.repeat(idx_conv.reshape(-1, 1), repeats=repeat_length, axis=1).flatten()
    loca_arr = loca_mat.flatten()
    repeat_length_arr = np.ones_like(loca_arr) * 1 / (repeat_length * 2)
    flip_length_arr = np.ones_like(loca_mat_flip) * 1 / 2
    idx_group = np.concatenate((idx_conv_trans_arr, idx_conv))
    loca_group = np.concatenate((loca_arr, loca_mat_flip))
    Pn_arr = np.concatenate((repeat_length_arr, flip_length_arr))
    Q_mat_tot = scipy.sparse.csr_matrix((Pn_arr, (idx_group, loca_group)))
    return np.array(Q_mat_tot[idx_conv][:, idx_conv].todense())

############################################## Uniform Proposal ####################################################
def get_Q_uniform_mat(idx_conv):
    """
    Construct a uniform proposal matrix Q for MCMC transitions.

    This function generates a uniform proposal probability matrix for the given set of state indices.
    Each entry Q[i, j] in the matrix is set to 1/length, where length is the total number of states,
    ensuring that the probability of proposing a transition from any state to any other state is the same.

    Parameters:
    -----------
    idx_conv : array_like
        An array of integers representing the current occupation number vector (ONV) configurations
        in their compact integer form.

    Returns:
    --------
    Q_mat_tot : np.ndarray
        A 2D numpy array of shape (length, length) representing the uniform proposal matrix,
        where each element is equal to 1/length.
    """
    length = len(idx_conv)
    Q_mat_tot = np.ones((length, length))/length
    return Q_mat_tot

############################################## Effective Proposal ##################################################
def get_Q_effective_mat(vv):
    """
    Construct an effective proposal matrix Q for MCMC based on eigenvectors.

    This function generates a symmetric proposal probability matrix Q where
    Q[i, j] represents the probability of proposing a transition from configuration i
    to configuration j.

    Parameters
    ----------
    vv : ndarray of shape (n_basis, n_states)
        Matrix of eigenvectors of the Hamiltonian. Each column corresponds to an eigenstate
        expressed in the computational basis (Fock space).

    Returns
    -------
    Q_mat : ndarray of shape (n_basis, n_basis)
        Symmetric proposal matrix Q, where Q[i, j] is the proposal probability
        from configuration i to configuration j.

    Notes
    -----
    The matrix elements are defined as:
        Q[i, j] = sum_n |vv[i, n]|^2 * |vv[j, n]|^2

    where the sum is over all eigenstates n. The result is a symmetric matrix
    that reflects the probability overlap of configurations across all eigenstates.
    This matrix is suitable for use in symmetric Metropolis-Hastings algorithms.
    """
    Pn_tot = (vv.conj() * vv).real
    Q_mat = np.einsum('in,jn -> ij', Pn_tot, Pn_tot, optimize=True)
    del Pn_tot
    return Q_mat

#################################################  Quantum Proposal ################################################
def get_Q_quantum_mat(ee, vv, time):
    """
    Construct the quantum walk transition probability matrix Q(t) based on eigenstates and time evolution.

    This function computes the time-dependent proposal matrix for a continuous-time quantum walk,
    where each element Q[i, j] gives the probability of transitioning from configuration i to j
    under unitary evolution generated by a Hamiltonian with eigenvalues `ee` and eigenvectors `vv`.

    Parameters
    ----------
    ee : ndarray of shape (n_states,)
        Eigenvalues of the Hamiltonian.

    vv : ndarray of shape (n_basis, n_states)
        Eigenvectors of the Hamiltonian. Each column is an eigenstate in the computational basis.

    time : float
        Time parameter for the unitary evolution.

    Returns
    -------
    Q_mat : ndarray of shape (n_basis, n_basis)
        The transition probability matrix Q(t), where Q[i, j] = |⟨i|U(t)|j⟩|^2.
        This matrix is symmetric and doubly stochastic (if the basis is complete).

    Notes
    -----
    The unitary evolution is given by:
        U(t) = vv @ diag(exp(-i * ee * t)) @ vv.T

    Then, the transition probability matrix is:
        Q[i, j] = |⟨i|U(t)|j⟩|^2

    This type of transition matrix can be used to define quantum-inspired or quantum-walk-based
    proposal mechanisms for MCMC algorithms.
    """
    Q_mat = np.abs((vv * np.exp(-1.j * ee * time)) @ vv.T) ** 2
    return Q_mat


####################################################################################################################
def get_special_space(x: int, sorb: int, noa: int, nob: int, device=None):
    """
    Generate a space of FCI states based on the provided number of orbitals and electrons.

    This function generates all (or part of) the Full Configuration Interaction (FCI) state space,
    where each state is represented as a binary vector of length `sorb` (number of spin orbitals).
    The configuration is built by choosing `noa` alpha electrons from the even-indexed orbitals and
    `nob` beta electrons from the odd-indexed orbitals, assuming that orbitals are arranged in an
    interleaved fashion (alpha, beta, alpha, beta, ...).

    Parameters:
    -----------
    x : int
        Total number of orbitals to consider. It must be even.
    sorb : int
        Total number of spin orbitals (should be equal to the length of the output binary vector).
    noa : int
        Number of occupied alpha orbitals (electrons).
    nob : int
        Number of occupied beta orbitals (electrons).
    device : optional
        An optional device parameter for potential GPU acceleration (currently unused).

    Returns:
    --------
    spins : np.ndarray, shape (n_configurations, sorb)
        A binary matrix where each row represents an FCI state. A '1' indicates that an orbital is occupied.

    Raises:
    -------
    AssertionError
        If `x` is not even, or if `x` is less than the total number of electrons (noa + nob).

    Notes:
    ------
    - The function uses itertools.combinations to generate all possible combinations of occupied orbitals.
    - Even indices (0, 2, 4, ...) correspond to alpha orbitals, while odd indices (1, 3, 5, ...) correspond to beta orbitals.
    """
    # Check that x is even and there are enough orbitals to hold all electrons.
    assert x % 2 == 0 and x <= sorb and x >= (noa + nob)

    # Generate all combinations of occupied alpha orbitals (even indices) and beta orbitals (odd indices).
    noA_lst = list(itertools.combinations([i for i in range(0, x, 2)], noa))
    noB_lst = list(itertools.combinations([i for i in range(1, x, 2)], nob))

    # Total number of configurations is the product of the number of alpha and beta combinations.
    m = len(noA_lst)
    n = len(noB_lst)
    spins = np.zeros((m * n, sorb), dtype=np.int64)

    # For each combination of alpha and beta orbitals, set the corresponding positions to 1.
    for i, lstA in enumerate(noA_lst):
        for j, lstB in enumerate(noB_lst):
            # Compute a unique index for the current configuration.
            idx = i * m + j
            spins[idx, lstA] = 1
            spins[idx, lstB] = 1
    return spins


def get_sigmaAidx(nqubits, noa, nob):
    """
    Generate a sorted array of integer indices for the FCI states.

    This function generates the FCI state space (using get_special_space) and then converts each binary
    configuration (state) into its corresponding integer representation using bit weights. The resulting
    integer indices uniquely represent the FCI states and are sorted in ascending order.

    Parameters:
    -----------
    nqubits : int
        Total number of spin orbitals.
    noa : int
        Number of occupied alpha orbitals (electrons).
    nob : int
        Number of occupied beta orbitals (electrons).

    Returns:
    --------
    idx : np.ndarray
        A sorted 1D array of integer representations of the FCI states.

    Notes:
    ------
    - The binary representation of each FCI state is converted to an integer by assigning a weight of 2^(position)
      to each bit, with the bits ordered from most significant to least significant.
    """
    # Generate the FCI configuration space as a binary matrix.
    CI_arr = get_special_space(nqubits, nqubits, noa, nob)

    # Create a weight vector for binary-to-integer conversion. The weight for each bit is 2^(position).
    two_vector = (2 ** np.arange(nqubits))[::-1]

    # Compute the integer index corresponding to each binary configuration.
    idx = (CI_arr.dot(two_vector))

    # Sort the indices in ascending order.
    idx.sort()
    return idx

###############################################################################################
def get_A(Pn):
    """
    Compute the acceptance probability matrix A based on a probability vector Pn.

    This function is used to calculate the acceptance rates in a Monte Carlo or sampling
    algorithm. The acceptance probability for a move from state i to state j is defined as
    A[i, j] = min(Pn[j]/Pn[i], 1), ensuring that the probability does not exceed 1.

    Parameters:
    -----------
    Pn : array_like
        A 1D array (or vector) of probabilities associated with each state.

    Returns:
    --------
    A : np.ndarray
        A 2D array (matrix) where each element A[i, j] represents the acceptance probability
        of transitioning from state i to state j, computed as:
              A[i, j] = min(Pn[j]/Pn[i], 1)
    """
    # Compute the outer product of the reciprocal of Pn and Pn.
    A_mat = np.einsum('i,j->ij', 1/Pn, Pn, optimize=True)
    # Create a matrix of ones of the same shape.
    ones_mat = np.ones((len(Pn), len(Pn)))
    # Set A[i, j] = min(A_mat[i, j], 1).
    A = np.where(A_mat < ones_mat, A_mat, ones_mat)
    return A

def get_Pij(Q, A):
    """
    Compute the MCMC transition probability matrix Pij.

    This function calculates the transition probabilities for an MCMC algorithm by first taking
    the element-wise product of two matrices, Q and A, where A is typically an acceptance probability
    matrix (computed by get_A) and Q is a proposal probability matrix. The resulting matrix is then
    adjusted by modifying its diagonal entries so that each row sums to 1, thereby ensuring a proper
    probability distribution.

    Parameters:
    -----------
    Q : np.ndarray
        A 2D array representing the proposal probability matrix, where each element Q[i, j] is the
        probability of proposing a transition from state i to state j.
    A : np.ndarray
        A 2D acceptance probability matrix, typically computed by get_A(Pn), where each element A[i, j]
        represents the acceptance probability of a proposed move from state i to state j.

    Returns:
    --------
    result : np.ndarray
        A 2D array representing the transition probability matrix Pij for the MCMC algorithm, where:
          - The off-diagonal elements are given by Q[i, j] * A[i, j].
          - The diagonal elements are adjusted such that each row sums to 1, ensuring that
            Pij is a valid probability distribution.
    """
    # Compute the element-wise product of Q and A to get the initial transition probabilities.
    result = Q * A
    length = result.shape[0]
    for i in range(length):
        # Temporarily set the diagonal element to 0.
        result[i, i] = 0
        # Set the diagonal element to 1 minus the sum of the off-diagonal elements in that row.
        result[i, i] = 1 - np.sum(result[i])
    return result

def sort_eevv(ee, vv):
    """
    Sort eigenvalues and eigenvectors in descending order of the absolute eigenvalue.

    This function takes a set of eigenvalues and eigenvectors, sorts them by the absolute value
    of the eigenvalues in descending order, and reorders the eigenvectors accordingly.

    Parameters:
    -----------
    ee : array_like
        A 1D array of eigenvalues.
    vv : array_like
        A 2D array where each column corresponds to an eigenvector associated with the eigenvalues in ee.

    Returns:
    --------
    sorted_eigenvalues : np.ndarray
        A 1D array of eigenvalues sorted in descending order (by absolute value).
    sorted_eigenvectors : np.ndarray
        A 2D array of eigenvectors reordered to correspond with the sorted eigenvalues.
    """
    # Take the absolute value of eigenvalues and get the sorting indices in descending order.
    ee = np.abs(ee)
    sorted_indices = np.argsort(ee)[::-1]
    # Sort eigenvalues and eigenvectors accordingly.
    sorted_eigenvalues = ee[sorted_indices]
    sorted_eigenvectors = vv[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvectors