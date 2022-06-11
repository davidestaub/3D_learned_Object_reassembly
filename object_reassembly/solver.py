import mosek, sys
import cvxpy as cp
import numpy as np
from cvxpy.atoms import norm, abs
from numpy.linalg import det


# from mip_cvxpy import PYTHON_MIP

def conv_SO3(x):
    x11 = x[1]
    x12 = x[2]
    x13 = x[3]
    x21 = x[4]
    x22 = x[5]
    x23 = x[6]
    x31 = x[7]
    x32 = x[8]
    x33 = x[9]

    L_tmp = [[x11 + x22 + x33, x32 - x23, x13 - x31, x21 - x12],
             [x32 - x23, x11 - x22 - x33, x21 + x12, x13 + x31],
             [x13 - x31, x21 + x12, x22 - x11 - x33, x32 + x23],
             [x21 - x12, x13 + x31, x32 + x23, x33 - x11 - x22]]

    L = list_to_cvx_mat(L_tmp)

    return L


def getQqp_simTransf(C, G, S=None):
    if S is None:
        S = np.eye(3)

    c1 = C[0]
    c2 = C[1]
    c3 = C[2]
    g1 = G[0]
    g2 = G[1]
    g3 = G[2]

    A = [[c1, c2, c3, 0, 0, 0, 0, 0, 0, 1, 0, 0, -g1],
         [0, 0, 0, c1, c2, c3, 0, 0, 0, 0, 1, 0, -g2],
         [0, 0, 0, 0, 0, 0, c1, c2, c3, 0, 0, 1, -g3]]

    At_A = np.matmul(np.transpose(A), A)
    q = At_A[12, 0:11] * 2
    p = At_A[12, 12]
    Q = At_A[0:11, 0:11] + 1e-15 * np.eye(11)

    return At_A, q, p, Q


def list_to_cvx_mat(p_list):
    rows = []
    for row in p_list:
        rows.append(cp.hstack([item for item in row]))
    return cp.vstack(rows)


def run_solver(U, V, th=0.5e-1, sLU=[0.9, 1.1], S=np.eye(3), enfRot=True, verbose=False, tol=1e-1):
    dim = 12
    N = U.shape[1]

    x = cp.Variable(dim)
    z = cp.Variable(N, boolean=True)

    s = cp.Variable(1)
    big_M = 10 ^ 5

    constraints = []

    # L constraint
    if enfRot:
        diag = np.diag([s, s, s, s])
        L = list_to_cvx_mat(diag) + conv_SO3(x)
        constraints.append(L >> 0)

    # L\infty constraints
    for i in range(N):
        M, _, _, _ = getQqp_simTransf(np.ravel(U[:, i]), np.ravel(V[:, i]), S)
        constraints.append(norm(M @ cp.hstack([x, 1])) <= th + z[i] * big_M)

    # s constraints (bounds)
    lower_S = sLU[0]
    uppder_S = sLU[1]

    constraints.append(s >= lower_S)
    constraints.append(s <= uppder_S)

    objective = cp.Minimize(sum(abs(z)))

    prob = cp.Problem(objective, constraints)

    print("Solving optimization...")

    prob.solve(solver=cp.MOSEK)

    if prob.status not in ["infeasible", "unbounded"]:

        print("Optimization successful!")

        R = np.reshape(x.value[:9], (3, 3))
        t = x.value[9:]
        s_opt = np.cbrt(np.abs(det(R)))
        inliers = [1 if i < tol else 0 for i in z.value]
        x_opt = x.value

        sol = {"R": R,
               "s_opt": s_opt,
               "t": t,
               "inliers": inliers,
               "x_opt": x_opt,
               "sol": prob.solution}

        return sol

    else:
        print("Optimization failed, problem " + str(prob.status) + " :(")
        return None


if __name__ == "__main__":
    U = np.eye(3)
    V = np.eye(3)
    run_solver(U, V)
