import numpy as np
import numpy.linalg as la

def get_ctrl(A : np.matrix, B : np.ndarray) -> np.matrix:
    Wc = np.empty(A.shape)
    for i in range(A.shape[0]):
        Wc[:,i] = la.matrix_power(A, i) @ B
    
    return Wc

def get_obsv(A : np.matrix, C : np.ndarray) -> np.matrix:
    Wo = np.empty(A.shape)
    for i in range(A.shape[0]):
        Wo[i,:] = C @ la.matrix_power(A, i)

    return Wo

def get_char_pol_array(poles : np.ndarray[complex]) -> np.ndarray:
    q = np.array([1, -poles[0]])

    for i in range(1, poles.shape[0]):
        q = np.convolve(q, [1, -poles[i]])

    return q.real

def apply_mat_on_pol(A : np.matrix, q : np.ndarray) -> np.matrix:
    Q = np.zeros(A.shape)

    for i in range(q.size):
        Q += q[q.size - 1 - i]*la.matrix_power(A, i)

    return Q

def get_Ghat_Hhat(G : np.matrix, H : np.ndarray) -> tuple[np.matrix, np.ndarray]:
    Ghat = np.empty((G.shape[0] + 1, G.shape[1] + 1))
    Ghat[:G.shape[0], :G.shape[1]] = G
    Ghat[:G.shape[0], -1         ] = H
    Ghat[-1          , :         ] = [0 for i in range(Ghat.shape[1])]

    Hhat = np.zeros((Ghat.shape[0],))
    Hhat[-1] = 1

    return Ghat, Hhat

def get_K_discrete(Wc : np.matrix, qc_G : np.matrix, G : np.matrix, H : np.ndarray, C : np.ndarray) -> np.ndarray:
    aux = np.zeros((Wc.shape[0],))
    aux[-1] = 1
    Khat = -aux@la.inv(Wc)@qc_G

    print(Khat)

    aux2 = np.empty((G.shape[0] + 1, G.shape[1] + 1))
    aux2[:G.shape[0], :G.shape[1]] = G - np.eye(G.shape[0])
    aux2[:G.shape[0], -1         ] = H
    aux2[-1         , :G.shape[1]] = C @ G
    aux2[-1         , -1         ] = C @ H

    Khat[-1] += 1

    return Khat@la.inv(aux2)

G = np.matrix([[1, 0.6321],
               [0, 0.3679]])

H = np.array([0.3679, 0.6321])

C = np.array([-2, 3])

Ghat, Hhat = get_Ghat_Hhat(G, H)
print("Ghat =\n", Ghat, "\n")
print("Hhat =\n", Hhat, "\n")

Wc = get_ctrl(Ghat, Hhat)
print("Wc =\n", Wc, "\n")

poles = np.array([complex(0.5, 0.2), complex(0.5, -0.2), -0.1])
qc = get_char_pol_array(poles)
qc_Ghat = apply_mat_on_pol(Ghat, qc)
print("qc(Ghat) =\n", qc_Ghat, "\n")

K = get_K_discrete(Wc, qc_Ghat, G, H, C)
print("K =", K, "\n")