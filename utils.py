import numpy as np
import numpy.linalg as la

# gera a matriz de controlabilidade dado A e B.
def get_ctrl(A : np.matrix, B : np.ndarray) -> np.matrix:
    Wc = np.empty(A.shape)
    for i in range(A.shape[0]):
        Wc[:,i] = la.matrix_power(A, i) @ B # Wc = [A AB A^2B ... A^(n-1)B]
    
    return Wc

# gera a matriz de observabilidade dado A e C
def get_obsv(A : np.matrix, C : np.ndarray) -> np.matrix:
    Wo = np.empty(A.shape)
    for i in range(A.shape[0]):
        Wo[i,:] = C @ la.matrix_power(A, i) # Wo = [A CA CA^2 ... CA^(n-1)]^T

    return Wo

# gera um array contendo os coeficientes do polinômio característico dados os polos
# ex.: 3s^2 - 4s + 2 é representado por [3, -4, 2]
def get_char_pol_array(poles : np.ndarray[complex]) -> np.ndarray:
    # cada polo é representado por um polinomio (s - p) <=> [1, -p]
    q = np.array([1, -poles[0]])

    # adquirir o array completo fazendo a multiplicacao de todos os polinomios.
    # isso é equivalente a fazer a convolução de todos os arrays de todos os polos
    for i in range(1, poles.shape[0]):
        q = np.convolve(q, [1, -poles[i]])

    return q.real

# dado o array do polinômio q(s), aplica-se A. Retorna q(A).
def apply_mat_on_pol(A : np.matrix, q : np.ndarray) -> np.matrix:
    Q = np.zeros(A.shape)

    for i in range(q.size):
        Q += q[q.size - 1 - i]*la.matrix_power(A, i)

    return Q

# a partir de matrizes G e H, constrói o sistema aumentado para o seguidor de referência discreto
def get_Ghat_Hhat(G : np.matrix, H : np.ndarray) -> tuple[np.matrix, np.ndarray]:
    Ghat = np.empty((G.shape[0] + 1, G.shape[1] + 1))
    # aplicação de fórmula
    Ghat[:G.shape[0], :G.shape[1]] = G
    Ghat[:G.shape[0], -1         ] = H
    Ghat[-1          , :         ] = [0 for i in range(Ghat.shape[1])]

    Hhat = np.zeros((Ghat.shape[0],))
    Hhat[-1] = 1

    return Ghat, Hhat

# A partir das matrizes A, B e C, constrói o sistema aumentado para o seguidor de referências contínuo
def get_Ahat_Bhat(A : np.matrix, B : np.ndarray, C: np.ndarray) -> tuple[np.matrix, np.ndarray]:
    Ahat = np.empty((A.shape[0] + B.size, A.shape[1] + 1))
    # Aplicação de fórmula
    Ahat[:          ,0 ] = 0
    Ahat[:A.shape[1],1:] = A
    Ahat[-1         , :] = C

    Bhat = np.zeros((B.size + 1,))
    Bhat[1:] = B

    return Ahat, Bhat

# A partir de Wc, q(G), G, H e C, retorna os ganhos para o seguidor de referencia discreto
def get_K_discrete(Wc : np.matrix, qc_G : np.matrix, G : np.matrix, H : np.ndarray, C : np.ndarray) -> np.ndarray:
    # aux = [0 0 0 ... 1]
    aux = np.zeros((Wc.shape[0],))
    aux[-1] = 1
    # K chapéu é -[0 0 0 ... 1]*Wc^-1*qc(G)
    Khat = -aux@la.inv(Wc)@qc_G

    print("Khat =", Khat, "\n")

    # aux2 é a matriz que precisa ser invertida, na formula do K para o seguidor de referencia
    aux2 = np.empty((G.shape[0] + 1, G.shape[1] + 1))
    aux2[:G.shape[0], :G.shape[1]] = G - np.eye(G.shape[0])
    aux2[:G.shape[0], -1         ] = H
    aux2[-1         , :G.shape[1]] = C @ G
    aux2[-1         , -1         ] = C @ H

    Khat[-1] += 1

    # K = (Khat + [0 0 ... 0 1])*aux2^-1
    return Khat@la.inv(aux2)

# a partir de Wc e qc(Ahat), retorna os ganhos para uma realimentação de estados QUALQUER exceto seguidor de referência discreto
def get_K_cont(Wc : np.matrix, qc_Ahat : np.matrix) -> np.ndarray:
    aux = np.zeros((Wc.shape[0],))
    aux[-1] = 1

    # K = -[0 0 ... 0 1]*Wc^-1*qc(Ahat)
    return -aux @ la.inv(Wc) @ qc_Ahat

### EXEMPLO PARA UM SEGUIDOR DE REFERÊNCIA DISCRETO.

# matriz G
G = np.matrix([[1, 0.6321],
               [0, 0.3679]])

# matriz H
H = np.array([0.3679, 0.6321])

# matriz C
C = np.array([-2, 3])

# gera G chapéu e H chapéu a partir de G e H.
Ghat, Hhat = get_Ghat_Hhat(G, H)
print("Ghat =\n", Ghat, "\n")
print("Hhat =\n", Hhat, "\n")

# retorna a matriz de controlabilidade a partir de G chapéu e H chapéu
Wc = get_ctrl(Ghat, Hhat)
print("Wc =\n", Wc, "\n")

# define os polos do seguidor
poles = np.array([complex(0.5, 0.2), complex(0.5, -0.2), -0.1])
# retorna o array do polinomio
qc = get_char_pol_array(poles)
# aplica a matriz G chapéu no polinômio
qc_Ghat = apply_mat_on_pol(Ghat, qc)
print("qc(Ghat) =\n", qc_Ghat, "\n")

# retorna os ganhos
K = get_K_discrete(Wc, qc_Ghat, G, H, C)
print("K =", K, "\n")