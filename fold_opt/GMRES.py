import torch
from torch.func import vmap

# An implementation of GMRes in torch
#    which is also vmap-ready
# For compatibility with vmap, it requires dummy inputs:
# H = torch.zeros(M + 1, M)
# Q = torch.zeros(M,b.shape[0])
# O = torch.eye(M)
# Where M is the number of GMRes iterations
def GMRes(A, b, x0, H, Q, O, M, tol=1e-6):

    I2 = torch.eye(2);
    I2skew = torch.zeros(2,2); I2skew[0,1] = 1.0; I2skew[1,0] = -1.0;

    r =  b - A@x0
    beta = torch.norm(r)
    Q[0] = r / torch.norm(r)

    e = torch.zeros(H.shape[0]); e[0] = 1.0
    for k in range(M-1):
        y = A@Q[k]
        for j in range(k+1):
            H[j][k] = Q[j]@y
            y = y - H[j][k]*Q[j]
        H[k+1][k] = torch.norm(y)

        Q[k+1] = torch.where( H[k+1][k] > tol, y / H[k+1][k], torch.zeros(y.shape) )

        if k==0:
            O_small,R = torch.linalg.qr(H[:k+2,:k+1], mode='complete'); O_small = O_small.T   #k=1: Q=1, R=[R_11]=[A_11] ?
            O[:O_small.shape[0],:O_small.shape[1]] = O_small
        else:
            # Create the Givens matrix and use it to update O
            rho_sigma = O[k:k+2,:k+2]@H[:k+2,k+1-1]
            Gblock = (rho_sigma[0]*I2 + rho_sigma[1]*I2skew) / torch.norm(rho_sigma)
            O[k:k+2,:] = Gblock@O[k:k+2,:]

            # Recover R from O and H
            R = O[:k+2,:k+2]@H[:k+2,:k+1]

            # Create the RHS of the least squares problem
            g_tild = beta * O[:k+2,:k+2]@e[:k+2]
            g = g_tild[:-1]

            # Least squares is equivalent to this back-substitution
            c = torch.linalg.solve_triangular(R[:g.shape[0]], g.unsqueeze(1), upper=True).squeeze() # TODO: go one shorter?
            x = (Q[:len(c)].T)@c + x0

    return x






# A modular version of the above GMRes function
# Meant to accomodate situations in which the matrix mutiplication y = A@Q[k]
#       cannot be vmapped
def GMRes_mod(A, b, x0, H, Q, O, M, tol=1e-8):

    r =  b - matrix_mult(A, x0)
    I2, I2skew, r, beta, Q, e = setup(r, b, x0, H, Q, O, M)

    for k in range(M-1):
        y = matrix_mult(A, Q[k])
        H,Q = update_H_Q(H, Q, y, k, tol)
        if k==0:
            # Initialize the O and R factors
            O, R = get_O_R(H, O, I2, I2skew, k)
        else:
            # Create the Givens matrix and use it to update O
            # Recover R from O and H
            O, R = update_O_R(H, O, I2, I2skew, k)
            # Create the RHS of the least squares problem
            # Least squares is equivalent to this back-substitution
            x = solve_x(Q,O,R,x0,e,beta,k)
    return x




def GMRes_mod_vec(A, b, x0, M, tol=1e-8):

    N = b.shape[1]
    B = b.shape[0]

    M = max(M,3)
    M = min(M,N+1)

    H = torch.zeros(M + 1, M).repeat(B,1,1)
    Q = torch.zeros(M,N).repeat(B,1,1)
    O = torch.eye(M).repeat(B,1,1)


    r =  b - v_matrix_mult(A, x0)
    I2, I2skew, r, beta, Q, e = v_setup(r, b, x0, H, Q, O, M)

    for k in range(M-1):
        #y = v_matrix_mult(A, Q[k])
        y = v_matrix_mult(A, Q[:,k,:])
        H,Q = v_update_H_Q(H, Q, y, k, tol)
        if k==0:
            # Initialize the O and R factors
            O, R = v_get_O_R(H, O, I2, I2skew, k)
        else:
            # Create the Givens matrix and use it to update O
            # Recover R from O and H
            O, R = v_update_O_R(H, O, I2, I2skew, k)
            # Create the RHS of the least squares problem
            # Least squares is equivalent to this back-substitution
            x = v_solve_x(Q,O,R,x0,e,beta,k)
    return x



def setup(r, b, x0, H, Q, O, M):
    I2 = torch.eye(2);
    I2skew = torch.zeros(2,2); I2skew[0,1] = 1.0; I2skew[1,0] = -1.0;
    beta = torch.norm(r)
    Q[0] = r / torch.norm(r)
    e = torch.zeros(H.shape[0]); e[0] = 1.0
    return I2, I2skew, r, beta, Q, e


# Used to implement the lines:
# r = b - A@x0
# y = A@Q[k]
def matrix_mult(A, z):
    y = A@z
    return y


def update_H_Q(H, Q, y, k, tol=1e-6):
    for j in range(k+1):
        H[j][k] = Q[j]@y
        y = y - H[j][k]*Q[j]
    H[k+1][k] = torch.norm(y)
    Q[k+1] = torch.where( H[k+1][k] > tol, y / H[k+1][k], torch.zeros(y.shape) )
    return H,Q


def get_O_R(H, O, I2, I2skew, k): # extra input k
    O_small,R = torch.linalg.qr(H[:k+2,:k+1], mode='complete'); O_small = O_small.T   #k=1: Q=1, R=[R_11]=[A_11] ?
    O[:O_small.shape[0],:O_small.shape[1]] = O_small
    return O, R


def update_O_R(H, O, I2, I2skew, k): # extra input k
    # Create the Givens matrix and use it to update O
    rho_sigma = O[k:k+2,:k+2]@H[:k+2,k+1-1]
    Gblock = (rho_sigma[0]*I2 + rho_sigma[1]*I2skew) / torch.norm(rho_sigma)
    O[k:k+2,:] = Gblock@O[k:k+2,:]

    # Recover R from O and H
    R = O[:k+2,:k+2]@H[:k+2,:k+1]
    return O, R


def solve_x(Q,O,R,x0,e,beta,k):
    # Create the RHS of the least squares problem
    g_tild = beta * O[:k+2,:k+2]@e[:k+2]
    g = g_tild[:-1]

    # Least squares is equivalent to this back-substitution
    c = torch.linalg.solve_triangular(R[:g.shape[0]], g.unsqueeze(1), upper=True).squeeze() # TODO: go one shorter?
    x = (Q[:len(c)].T)@c + x0
    return x

v_setup       = vmap(setup,       in_dims = (0,0,0,0,0,0,None))  #inputs: (r, b, x0, H, Q, O, M)
v_matrix_mult = vmap(matrix_mult, in_dims = (0,0))               #inputs: (A, z)
v_update_H_Q  = vmap(update_H_Q,  in_dims = (0,0,0,None,None))   #inputs: (H, Q, y, k, tol=1e-6)
v_get_O_R     = vmap(get_O_R,     in_dims = (0,0,0,0,None))      #inputs: (H, O, I2, I2skew, k)
v_update_O_R  = vmap(update_O_R,  in_dims = (0,0,0,0,None))      #inputs: (H, O, I2, I2skew, k)
v_solve_x     = vmap(solve_x,     in_dims = (0,0,0,0,0,0,None))  #inputs: (Q,O,R,x0,e,beta,k)
