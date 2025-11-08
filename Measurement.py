
import numpy as np

def pauli_op_mat_from_str(op_str):
    '''return the matrix of the pauli operator.'''
    pauli_matrice = [np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]]),    np.array([[0, -1j], [1j, 0]]),     np.array([[1, 0], [0, -1]])]
    str_map = {'I':0,'X': 1, 'Y':2, "Z":3}
    # pauli_matrice = [np.array([[0, 1], [1, 0]]),    np.array([[0, -1j], [1j, 0]]),     np.array([[1, 0], [0, -1]])]
    # str_map = {'X': 0, 'Y':1, "Z":1}

    return pauli_matrice[str_map[op_str]]

def get_pauli_str_matrix(pauli_string):
    '''Given a pauli string, return the matrix corresponding to this operator.
    '''
    if len(pauli_string) < 1:
        raise ValueError("Empty pauli string.")
    mat = pauli_op_mat_from_str(pauli_string[0])
    
    if len(pauli_string) == 1:
        return mat
    
    for pauli_op_str in pauli_string[1:]:
        new_op_mat = pauli_op_mat_from_str(pauli_op_str)
        mat = np.kron(mat, new_op_mat)
    return mat

def gen_all(num_qubits: int) -> list[list[str]]:  
    """  
    Generate all possible combinations of Pauli operators for a given number of qubits.  
      
    Parameters:  
    - num_qubits: int, the number of qubits.  
      
    Returns:  
    - all_ps: List[List[str]], a list of lists representing all Pauli operator combinations.  
    """  
    pauli_ops = [ 'I','X', 'Y', 'Z']  
    # pauli_ops = [ 'X', 'Y', 'Z']  
    if num_qubits == 1:  
        return [[op] for op in pauli_ops]  
    else:  
        all_ps = []  
        Ps_prev = gen_all(num_qubits - 1)  
          
        for pi in pauli_ops:  
            for pj in Ps_prev:  
                # Create a new list by appending the current Pauli operator to the previous combination  
                all_ps.append([pi] + pj)  
          
        return all_ps  
    
def Bases_measure(num_qubits):
    pauli_string = gen_all(num_qubits)
    print(pauli_string )
    all_mat = []
    for ps in  pauli_string  :
        mat = get_pauli_str_matrix(ps)
        all_mat.append(mat)
    return  all_mat


# num_qubits =3
# #print(len(gen_all(num_qubits)))
# print(pauli_bases(num_qubits))
