import numpy as np
from pyscf import ao2mo
from mindquantum.algorithm.nisq.chem.transform import Transform
from mindquantum.third_party.interaction_operator import InteractionOperator
from mindquantum.core.operators import FermionOperator
from mindquantum.core.operators.polynomial_tensor import PolynomialTensor

#################### some molecular structures in 3D Cartesian coordinates ###################
def genH2O(bond=2.0,angle=104.5):
    """
    Generate the molecular structure of a water (H₂O) molecule in 3D Cartesian coordinates.

    Parameters:
    ----------
    bond : float, optional
        The O–H bond length in angstroms. Default is 2.0 Å.

    angle : float, optional
        The H–O–H bond angle in degrees. Default is 104.5°.

    Returns:
    -------
    structure : list of [str, list of float]
        A list of atomic symbols and their corresponding Cartesian coordinates in Ångström.
        The format is:
            [
                ['O', [x1, y1, z1]],
                ['H', [x2, y2, z2]],
                ['H', [x3, y3, z3]],
            ]
    """
    return [['O', [0, 0, 0]],
           ['H', [bond*np.sin(angle*np.pi/2/180),bond*np.cos(angle*np.pi/2/180), 0]],
           ['H', [-bond*np.sin(angle*np.pi/2/180), bond*np.cos(angle*np.pi/2/180), 0]]]

################################# construct Hamiltonian ####################################
def unitiy_mo_coeff_format(cm):
    # unity RHF UHF mo_coeff format
    if len(cm.shape) == 3:
        return cm
    elif len(cm.shape) == 2:
        return (cm, cm)


def get_ao2so_h1_h2_int(mo_coeff, h1e, h2e):
    """
    Transform one- and two-electron integrals from the atomic orbital (AO) basis
    to the spin-orbital (SO) basis in ABAB order.

    Parameters:
    ----------
    mo_coeff : ndarray or tuple of ndarrays
        Molecular orbital coefficients, typically obtained from SCF calculations (e.g., mf.mo_coeff).
        For non-spin-polarized systems, this can be a single 2D array.
        For spin-polarized systems, this should be a tuple of two arrays (alpha and beta components).

    h1e : ndarray
        One-electron Hamiltonian integrals in the AO basis. Shape: (nbas, nbas).

    h2e : ndarray
        Two-electron repulsion integrals in the AO basis. This should be a 4-index tensor in physicist’s notation (ij|kl),
        typically from ERI computations. Shape: (nbas, nbas, nbas, nbas).

    Returns:
    -------
    h1 : ndarray
        One-electron integrals in the spin-orbital basis with ABAB ordering.
        Shape: (2*nbas, 2*nbas). All alpha-beta and beta-alpha components are explicitly set to zero.

    h2 : ndarray
        Two-electron integrals in the spin-orbital basis with ABAB ordering.
        Shape: (2*nbas, 2*nbas, 2*nbas, 2*nbas). Mixed spin components (e.g., alpha-beta) are set to zero.
    """
    # input mo_coeff = mf.mo_coeff
    mo_coeff = unitiy_mo_coeff_format(mo_coeff)
    nbas = h1e.shape[0]
    b = np.zeros((nbas, 2 * nbas))
    b[:, ::2] = mo_coeff[0].copy()
    b[:, 1::2] = mo_coeff[1].copy()
    # INT1e:
    hcore = h1e
    h1 = b.T.dot(hcore).dot(b)
    h1[::2, 1::2] = h1[1::2, ::2] = 0.
    # INT2e:
    mf_eri = h2e
    h2 = ao2mo.general(mf_eri, (b, b, b, b), compact=False).reshape(2 * nbas, 2 * nbas, 2 * nbas, 2 * nbas)
    h2[::2, 1::2, :, :] = h2[1::2, ::2, :, :] = h2[:, :, ::2, 1::2] = h2[:, :, 1::2, ::2] = 0.
    return h1, h2

def get_fermion_operator(operator):
    """
    Convert a `PolynomialTensor` to a `FermionOperator`.

    Parameters:
    ----------
    operator : PolynomialTensor
        A `PolynomialTensor` representing the second-quantized Hamiltonian or observable.

    Returns:
    -------
    fermion_operator : FermionOperator
        The corresponding `FermionOperator` representation.
    """
    fermion_operator = FermionOperator()

    if isinstance(operator, PolynomialTensor):
        for term in operator:
            fermion_operator += FermionOperator(term, operator[term])
        return fermion_operator

    raise TypeError("Unsupported type of oeprator {}".format(operator))

def get_fermion_hamiltonian(ecore,h1,h2):
    """
    Construct the second-quantized molecular Fermionic Hamiltonian in the spin-orbital basis.

    Parameters:
    ----------
    ecore : float
        The nuclear repulsion energy (core energy) of the molecule.

    h1 : ndarray
        One-electron integrals in the spin-orbital basis. Shape: (n, n)

    h2 : ndarray
        Two-electron integrals in the spin-orbital basis. Shape: (n, n, n, n)

    Returns:
    -------
    fermion_hamiltonian : FermionOperator
        The Fermionic Hamiltonian expressed as a `FermionOperator` object.
    """
    h2 = h2.transpose(0,2,3,1)
    fermion_hamiltonian = InteractionOperator(ecore,h1,0.5*h2)
    fermion_hamiltonian = get_fermion_operator(fermion_hamiltonian)
    return fermion_hamiltonian

def get_qubit_operator(fermion_operator,fermion_transform):
    """
    Transform a FermionOperator into a QubitOperator (sum of Pauli strings) using a specified mapping.

    Parameters:
    ----------
    fermion_operator : FermionOperator
        The fermionic Hamiltonian or operator expressed in second quantization.

    fermion_transform : str
        The name of the transformation method. Supported values are:
            - 'jordan_wigner'
            - 'parity'
            - 'bravyi_kitaev'
            - 'bravyi_kitaev_tree'

    Returns:
    -------
    qubit_operator : QubitOperator
        The transformed operator represented as a sum of Pauli strings acting on qubits.
    """
    if fermion_transform == 'jordan_wigner':
        qubit_operator = Transform(fermion_operator).jordan_wigner()
    elif fermion_transform == 'parity':
        qubit_operator = Transform(fermion_operator).parity()
    elif fermion_transform == 'bravyi_kitaev':
        qubit_operator = Transform(fermion_operator).bravyi_kitaev()
    elif fermion_transform == 'bravyi_kitaev_tree':
        qubit_operator = Transform(fermion_operator).bravyi_kitaev_tree()
    return qubit_operator

def get_qubit_hamiltonian(ecore,h1,h2,fermion_transform):
    """
    Construct the qubit Hamiltonian of a molecule from its spin-orbital integrals.

    This function takes the molecular integrals in the spin-orbital basis, builds the corresponding
    fermionic Hamiltonian, and then maps it to a qubit Hamiltonian using a specified fermion-to-qubit
    transformation (e.g., Jordan-Wigner, Bravyi-Kitaev).

    Parameters:
    ----------
    ecore : float
        The nuclear repulsion energy (core energy) of the molecule.

    h1 : ndarray
        One-electron integrals in the spin-orbital basis. Shape: (n, n)

    h2 : ndarray
        Two-electron integrals in the spin-orbital basis. Shape: (n, n, n, n)

    fermion_transform : str
        The transformation method used to map fermionic operators to qubit operators.
        Supported options:
            - 'jordan_wigner'
            - 'parity'
            - 'bravyi_kitaev'
            - 'bravyi_kitaev_tree'

    Returns:
    -------
    qubit_hamiltonian : QubitOperator
        The molecular Hamiltonian expressed as a qubit operator (a sum of Pauli strings).
    """
    fermion_hamiltonian = get_fermion_hamiltonian(ecore,h1,h2)
    qubit_hamiltonian = get_qubit_operator(fermion_hamiltonian,fermion_transform)
    return qubit_hamiltonian

def generalop2pauliobAlocal_nk(qubit_op, nqubits):
    """
    Convert a QubitOperator into Pauli strings and coefficients for NetKet operator construction.

    This function parses a `QubitOperator` term-by-term and maps it to a string representation
    (e.g., 'XIZY') with identity operators ('I') inserted for qubits not involved in the term.

    Parameters:
    ----------
    qubit_op : QubitOperator
        A qubit Hamiltonian expressed as a sum of Pauli operators acting on multiple qubits.

    nqubits : int
        The total number of qubits. Determines the length of each Pauli string.

    Returns:
    -------
    ob_string_list : list of str
        A list of Pauli strings (length `nqubits`), each representing one term in the operator.
        For example: ['XII', 'IZX', 'YZY'].

    coeff_list : list of complex or float
        The coefficients corresponding to each Pauli string, taken from the original `QubitOperator`.
    """
    ob_string_list = []
    coeff_list = []
    mapping_dict = {}
    op2num_map = {'X': 0, 'Y': 1, 'Z': 2}
    num2op_map = ['X','Y','Z','I']
    for seq, (key, value) in enumerate(qubit_op.terms.items()):
        loca = []
        obs = []
        #coeff_list.append(value.const)
        coeff_list.append(value.const)
        mapping_dict[key] = seq
        for single_pauli in key:
            loca.append(single_pauli[0])
            obs.append(op2num_map[single_pauli[1]])
        op_string = ''
        count = 0
        for i in range(nqubits):
            if i in loca:
                op_string += num2op_map[obs[count]]
                count += 1
            else:
                op_string += 'I'
        ob_string_list.append(op_string)
    return ob_string_list, coeff_list