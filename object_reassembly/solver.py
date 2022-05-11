import mosek
import cvxpy as cp
import numpy as np

def conv_SO3(x):
    x11 = x[1]; x12 = x[2]; x13 = x[3]
    x21 = x[4]; x22 = x[5]; x23 = x[6]
    x31 = x[7]; x32 = x[8]; x33 = x[9]

    L = [[x11 + x22 + x33, x32 - x23, x13 - x31, x21 - x12],
    [x32 - x23, x11 - x22 - x33, x21 + x12, x13 + x31],
    [x13 - x31, x21 + x12, x22 - x11 - x33, x32 + x23],
    [x21 - x12, x13 + x31, x32 + x23, x33 - x11 - x22]]

    return L

def getQqp_simTransf(C, G, S=None):
    if not S:
        S = np.eye(3)

    c1 = C[1]; c2 = C[2]; c3 = C[3]
    g1 = G[1]; g2 = G[2]; g3 = G[3];

    A = [[c1, c2, c3, 0, 0, 0, 0, 0, 0, 1, 0, 0, -g1],
         [0, 0, 0, c1, c2, c3, 0, 0, 0, 0, 1, 0, -g2],
         [0, 0, 0, 0, 0, 0, c1, c2, c3, 0, 0, 1, -g3]]

    At_A = np.matmul(np.transpose(A), A)
    q = At_A[13, 1:12]*2
    p = At_A[13,13]
    Q = At_A[1:12, 1:12] + 10^(-15)*np.eye(12)

    return A, q, p, Q

def run_solver(U, V, th=0.5e-1, sLU=[0.9, 1.1], S=np.eye(3), enfRot=True):
    dim = 12;
    N = U.shape[1]

    x = cp.Variable((dim, 1))
    z = cp.Variable((N, 1))

    s = cp.Variable(1)
    big_M = 10^5

    constraints = []

    # L constraint
    if enfRot:
        L = s*np.eye(4) + conv_SO3(x)
        constraints.append(L >= 0)

    # L\infty constraints
    for i in range(N):
        M = getQqp_simTransf(U[:,i], V[:,i], S)
        constraints.append(np.linalg.norm(np.matmul(M, [x,1])) <= th + z[i]*big_M)

    # s constraints (bounds)
    lower_S = sLU[0];
    uppder_S = sLU[2];

    objective = np.sum(z)

    prob = cp.Problem(objective, constraints)
    result = prob.solve()





if __name__=="__main__":
    U = np.eye(3)
    V = np.eye(3)
    run_solver(U, V)