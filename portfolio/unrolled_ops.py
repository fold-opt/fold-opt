import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from qpth.qp import QPFunction

from qpth_plus.qpth.qp import QPFunction_Plus




# 1013 reworked from SQP_c
# The main intended difference is that this version
#   will receive initial starting values for primal and dual solution
# This reflects the main intended usage as a fixed point iterator
def DiffSQP(c, N, M_in, M_eq, g, e, grad_g, grad_e, grad_f, hess_L, alpha, max_iter, primal, dual ):

    batsize = c.shape[0]

    x   = primal
    lam = dual

    #print("in DiffSQP:")
    #print("dual = ")
    #print( dual )

    assert batsize==len(primal) and len(primal)==len(dual)

    # Ax == b
    # Gx <= h
    for _ in range(max_iter):

        #print("x = " )
        #print( x )
        #print("lam = " )
        #print( lam )


        G = torch.stack( [grad_g(x_i).double() for x_i in x] )
        h = torch.stack( [-g(x_i).double() for x_i in x] )#.unsqueeze(1)
        if e != None:
            A = torch.stack( [grad_e(x_i).double() for x_i in x] )
            b = torch.stack( [-e(x_i).double() for x_i in x] )
        else:
            A = torch.Tensor([])
            b = torch.Tensor([])

        Q = torch.stack( [hess_L(x[i],lam[i],c[i] ).double()    for i in range(len(x))] )
        p = torch.stack( [grad_f(x_i,c[i]).double() for i,x_i in enumerate(x)] )

        primaldual = QPFunction_Plus(verbose=-1,eps=1e-14,maxIter=1000)(Q,p,G,h,A,b)   #The Hessian is H=2Q, the SQP minimizes (1/2)xT H x, Amos puts in (1/2) automatically

        #d   = primaldual[ :,:N ]
        #lam = primaldual[ :, N:N+M_in ]
        #nu  = primaldual[ :,   N+M_in:N+M_in+M_eq ]
        #x = x+alpha*d



        d     = primaldual[ :,:N ]
        lam_d = primaldual[ :, N:N+M_in ]
        nu_d  = primaldual[ :,   N+M_in:N+M_in+M_eq ]
        x = x+alpha*d

        print("in DiffSQP:")
        print("lam = ")
        print( lam )
        print("lam_d = ")
        print( lam_d )

        lam = lam + alpha*(lam_d - lam)

    return x



# Adapted from DiffPGD and DiffSQP
# take same inputs as DiffSQP
# Form linear approximation to the constraints as in DiffSQP
#   then perform DiffPGD
def DiffPGDLinear(c, N, M_in, M_eq, g, e, grad_g, grad_e, grad_f, alpha, max_iter, primal_sol ):

    batsize = c.shape[0]

    x = primal_sol
    Q = torch.eye(N).repeat(batsize,1,1)


    assert batsize==len(primal_sol)

    # Ax == b
    # Gx <= h
    for _ in range(max_iter):

        #print("x = ")
        #print( x    )
        #input("waiting in DiffPGDLinear")


        G = torch.stack( [grad_g(x_i).double() for x_i in x] )
        h = torch.stack( [-g(x_i).double() for x_i in x] )#.unsqueeze(1)
        if e != None:
            A = torch.stack( [grad_e(x_i).double() for x_i in x] )
            b = torch.stack( [-e(x_i).double() for x_i in x] )
        else:
            A = torch.Tensor([])
            b = torch.Tensor([])


        grad = torch.stack( [grad_f(x_i,c[i].double()).double() for i,x_i in enumerate(x)] )
        x = x - alpha*grad

        x = QPFunction(verbose=-1,eps=1e-14,maxIter=1000)(2*Q.double(),-2*x,G.double(),h.double(),A.double(),b.double())   #The Hessian is H=2Q, the SQP minimizes (1/2)xT H x, Amos puts in (1/2) automatically


    return x


# Differentiably solve the following problem
# by unfolding projected gradient descent
# min_x f(x,c)
# s.t.
# A x == b
# G x <= h

# Inputs:
# c : the optimization problem parameters
# N : the number of optimization variables
# grad_f : a function for the gradient of f w.r.t x as a functino of x and c
# A,b,G,h : tensors defining the constraints
# max_iter : number of PGD iterations
# primal_sol : the initial iterate x_0
def DiffPGD(c, N, grad_f, G, h, A, b, alpha, max_iter, primal_sol):

    batsize = c.shape[0]
    x = primal_sol
    Q = torch.eye(N)

    for _ in range(max_iter):

        grad = torch.stack( [grad_f(x_i,c[i].double()).double() for i,x_i in enumerate(x)] )
        x = x - alpha*grad

        x = QPFunction(verbose=-1,eps=1e-10,maxIter=1000)(2*Q.double(),-2*x,G.double(),h.double(),A.double(),b.double())   #The Hessian is H=2Q, the SQP minimizes (1/2)xT H x, Amos puts in (1/2) automatically

    return x










class FixedPtPGD(nn.Module):
    def __init__(self, N, grad_f, G, h, A, b, alpha, max_iter, cvxmodule_blank):
        super(FixedPtPGD, self).__init__()
        self.cvx_blank = cvxmodule_blank #BlankFunctionWrapper( cvxmodule )
        self.N        = N
        self.grad_f   = grad_f
        self.G        = G
        self.h        = h
        self.A        = A
        self.b        = b
        self.alpha    = alpha
        self.max_iter = max_iter

    def forward(self,c):
        N        = self.N
        grad_f   = self.grad_f
        G        = self.G
        h        = self.h
        A        = self.A
        b        = self.b
        alpha    = self.alpha
        max_iter = self.max_iter

        batsize = c.shape[0]

        # wrap this in torch.no_grad()?
        x = self.cvx_blank(c)

        Q = torch.eye(N)

        # Ax == b
        # Gx <= h
        for _ in range(max_iter):

            grad = torch.stack( [grad_f(x_i,c[i].double()).double() for i,x_i in enumerate(x)] )
            x = x - alpha*grad
            print("x = ")
            print( x )
            x = QPFunction(verbose=-1,eps=1e-14,maxIter=1000)(2*Q.double(),-2*x,G.double(),h.double(),A.double(),b.double())   #The Hessian is H=2Q, the SQP minimizes (1/2)xT H x, Amos puts in (1/2) automatically
            print("x_proj = ")
            print( x )
        return x







# Is there a ADMM version of this PGD?
def PGD_linear(c, N, grad_f, G, h, A, b, alpha, max_iter ):

    # x is a batch
    x = torch.ones(len(c),N).double()#.unsqueeze(0).double()
    Q = torch.eye(N).double()

    # Ax == bs
    # Gx <= h
    for _ in range(max_iter):

        grad = torch.stack( [grad_f(x_i,c[i].double()).double() for i,x_i in enumerate(x)] )

        x = x - alpha*grad
        x = QPFunction(verbose=-1)(2*Q.double(),-2*x,G.double(),h.double(),A.double(),b.double())   #The Hessian is H=2Q, the SQP minimizes (1/2)xT H x, Amos puts in (1/2) automatically

    return x


def PGD_unrolled(c, N, grad_f, G, h, A, b, alpha, max_iter_sqp, max_iter_admm ):

    # x is a batch
    x = torch.ones(len(c),N).double()#.unsqueeze(0).double()
    Q = torch.eye(N).double()

    Q = torch.stack([Q for _ in range(len(x))])
    G = torch.stack([G for _ in range(len(x))])
    h = torch.stack([h for _ in range(len(x))])
    A = torch.stack([A for _ in range(len(x))])
    b = torch.stack([b for _ in range(len(x))])


    # Ax == b
    # Gx <= h
    rho = 1.0
    for _ in range(max_iter_sqp):

        grad = torch.stack( [grad_f(x_i,c[i]).double() for i,x_i in enumerate(x)] )

        x = x - alpha*grad
        #x = QPFunction(verbose=-1)(Q,-2*x,G,h,A,b)   #The Hessian is H=2Q, the SQP minimizes (1/2)xT H x, Amos puts in (1/2) automatically
        x = ADMM_qp(Q,-2*x,G,h,A,b,rho,max_iter_admm)[:,:N]


    return x




def SQP_ADMM(c, N, M_in, M_eq, g, e, grad_g, grad_e, grad_f, hess_L, alpha, max_iter, rho, max_iter_admm ):

    batsize = c.shape[0]

    #x   = torch.ones(  batsize, N ).double()
    #lam = torch.zeros( batsize, M_in ).double()

    #torch.manual_seed(15)
    x   = 0.5*torch.ones(  batsize, N ).double()
    lam = torch.zeros( batsize, M_in ).double()


    # Ax == b
    # Gx <= h
    for _ in range(max_iter):

        G = torch.stack( [grad_g(x_i).double() for x_i in x] )
        h = torch.stack( [-g(x_i).double() for x_i in x] )#.unsqueeze(1)
        if e != None:
            A = torch.stack( [grad_e(x_i).double() for x_i in x] )
            b = torch.stack( [-e(x_i).double() for x_i in x] )
        else:
            A = torch.Tensor([])
            b = torch.Tensor([])

        Q = torch.stack( [hess_L(x[i],lam[i],c[i] ).double()    for i in range(len(x))] )
        p = torch.stack( [grad_f(x_i,c[i]).double() for i,x_i in enumerate(x)] )

        primaldual = QPFunction_Plus(verbose=-1)(Q,p,G,h,A,b)   #The Hessian is H=2Q, the SQP minimizes (1/2)xT H x, Amos puts in (1/2) automatically

        d = ADMM_qp(Q,p,G,h,A,b,rho,max_iter_admm)[:,:N]
        d_amos = primaldual[ :,:N ]

        with torch.no_grad():
            print("ADMM error: {}".format(torch.nn.MSELoss()(d,d_amos)))

        """
        print("d = ")
        print( d    )
        print("d_amos = ")
        print( d_amos    )
        """

        #d    = primaldual[ :,:N ]
        lam_d = primaldual[ :, N:N+M_in ]
        nu_d  = primaldual[ :,   N+M_in:N+M_in+M_eq ]
        x   = x+alpha*d
        lam = lam + alpha*(lam_d - lam)

    return x




# 0810 Reworked from SQP_c
# Use SQP 'offline' to find a solution,
# then use gradients of the
# QP expansion around it
class FixedPtSQP(nn.Module):
    def __init__(self, N, M_in, M_eq, g, e, grad_g, grad_e, grad_f, hess_L, alpha, max_iter ):

        super(FixedPtSQP, self).__init__()
        BlankSQP  = BlankSQPWrapper( N, M_in, M_eq, g, e, grad_g, grad_e, grad_f, hess_L, alpha, max_iter )
        self.blank_SQP = BlankSQP

        self.N      = N
        self.grad_g = grad_g
        self.g      = g
        self.grad_e = grad_e
        self.e      = e
        self.hess_L = hess_L
        self.grad_f = grad_f

    def forward(self,c):
        N      = self.N
        N      = self.N
        grad_g = self.grad_g
        g      = self.g
        grad_e = self.grad_e
        e      = self.e
        hess_L = self.hess_L
        grad_f = self.grad_f

        batsize = c.shape[0]

        x_lam = self.blank_SQP(c)
        x   = x_lam[:,:N]
        lam = x_lam[:, N:]

        G = torch.stack( [grad_g(x_i).double() for x_i in x] )
        h = torch.stack( [-g(x_i).double() for x_i in x] )#.unsqueeze(1)
        if e != None:
            A = torch.stack( [grad_e(x_i).double() for x_i in x] )
            b = torch.stack( [-e(x_i).double() for x_i in x] )
        else:
            A = torch.Tensor([])
            b = torch.Tensor([])

        Q = torch.stack( [hess_L(x[i],lam[i],c[i] ).double()    for i in range(len(x))] )
        p = torch.stack( [grad_f(x_i,c[i]).double() for i,x_i in enumerate(x)] )

        primaldual = QPFunction_Plus(verbose=-1)(Q,p,G,h,A,b)
        d     = primaldual[ :,:N ]

        return x + d





# Same as fixed pt SQP, but
# the constraints of the QP approximation are omitted
class FixedPtSQPUnconstr(nn.Module):
    def __init__(self, N, M_in, M_eq, g, e, grad_g, grad_e, grad_f, hess_L, alpha, max_iter ):

        super(FixedPtSQPUnconstr, self).__init__()
        BlankSQP  = BlankSQPWrapper( N, M_in, M_eq, g, e, grad_g, grad_e, grad_f, hess_L, alpha, max_iter )
        self.blank_SQP = BlankSQP

        self.N      = N
        self.grad_g = grad_g
        self.g      = g
        self.grad_e = grad_e
        self.e      = e
        self.hess_L = hess_L
        self.grad_f = grad_f

    def forward(self,c):
        N      = self.N
        N      = self.N
        grad_g = self.grad_g
        g      = self.g
        grad_e = self.grad_e
        e      = self.e
        hess_L = self.hess_L
        grad_f = self.grad_f

        batsize = c.shape[0]

        x_lam = self.blank_SQP(c)
        x   = x_lam[:,:N]
        lam = x_lam[:, N:]

        #G = torch.stack( [grad_g(x_i).double() for x_i in x] )
        #h = torch.stack( [-g(x_i).double() for x_i in x] )#.unsqueeze(1)
        #if e != None:
        #    A = torch.stack( [grad_e(x_i).double() for x_i in x] )
        #    b = torch.stack( [-e(x_i).double() for x_i in x] )
        #else:
        #    A = torch.Tensor([])
        #    b = torch.Tensor([])

        Q = torch.stack( [hess_L(x[i],lam[i],c[i] ).double()    for i in range(len(x))] )
        p = torch.stack( [grad_f(x_i,c[i]).double() for i,x_i in enumerate(x)] )

        primaldual = QPFunction_Plus(verbose=-1)(Q,p,torch.Tensor([]),torch.Tensor([]),torch.Tensor([]),torch.Tensor([]))
        d     = primaldual[ :,:N ]

        return x + d











# 0817 reworked from FixedPtSQP
# Takes a differentiable QP from x* (computed offline) as in FixedPtSQP
# But then, takes several more differentiable steps
# Note - each step should be 0
# But does the gradient improve?
class FixedPtSQP_Plus(nn.Module):
    def __init__(self, N, M_in, M_eq, g, e, grad_g, grad_e, grad_f, hess_L, alpha, max_iter1, max_iter2 ):

        super(FixedPtSQP_Plus, self).__init__()
        BlankSQP  = BlankSQPWrapper( N, M_in, M_eq, g, e, grad_g, grad_e, grad_f, hess_L, alpha, max_iter1 )
        self.blank_SQP = BlankSQP

        self.N      = N
        self.M_in   = M_in
        self.M_eq   = M_eq
        self.grad_g = grad_g
        self.g      = g
        self.grad_e = grad_e
        self.e      = e
        self.hess_L = hess_L
        self.grad_f = grad_f
        self.max_iter1 = max_iter1
        self.max_iter2 = max_iter2
        self.alpha    = alpha

    def forward(self,c):
        N      = self.N
        M_in   = self.M_in
        M_eq   = self.M_eq
        grad_g = self.grad_g
        g      = self.g
        grad_e = self.grad_e
        e      = self.e
        hess_L = self.hess_L
        grad_f = self.grad_f
        max_iter2 = self.max_iter2
        alpha     = self.alpha

        batsize = c.shape[0]

        x_lam = self.blank_SQP(c)
        x   = x_lam[:,:N]
        lam = x_lam[:, N:]

        # Ax == b
        # Gx <= h
        for _ in range(max_iter2):

            G = torch.stack( [grad_g(x_i).double() for x_i in x] )
            h = torch.stack( [-g(x_i).double() for x_i in x] )#.unsqueeze(1)
            if e != None:
                A = torch.stack( [grad_e(x_i).double() for x_i in x] )
                b = torch.stack( [-e(x_i).double() for x_i in x] )
            else:
                A = torch.Tensor([])
                b = torch.Tensor([])

            Q = torch.stack( [hess_L(x[i],lam[i],c[i] ).double()    for i in range(len(x))] )
            p = torch.stack( [grad_f(x_i,c[i]).double() for i,x_i in enumerate(x)] )
            primaldual = QPFunction_Plus(verbose=-1)(Q,p,G,h,A,b)   #The Hessian is H=2Q, the SQP minimizes (1/2)xT H x, Amos puts in (1/2) automatically

            d     = primaldual[ :,:N ]
            lam_d = primaldual[ :, N:N+M_in ]
            nu_d  = primaldual[ :,   N+M_in:N+M_in+M_eq ]
            x = x+alpha*d
            lam = lam + alpha*(lam_d - lam)

        return x + d







# 0810 same as SQP_c but returns the last QP dual solution
# Intended for use in SQP_local
def SQP_c_dual(c, N, M_in, M_eq, g, e, grad_g, grad_e, grad_f, hess_L, alpha, max_iter ):

    batsize = c.shape[0]

    x   = 0.5*torch.ones(  batsize, N ).double()
    lam = torch.zeros( batsize, M_in ).double()


    # Ax == b
    # Gx <= h
    for _ in range(max_iter):

        G = torch.stack( [grad_g(x_i).double() for x_i in x] )
        h = torch.stack( [-g(x_i).double() for x_i in x] )#.unsqueeze(1)
        if e != None:
            A = torch.stack( [grad_e(x_i).double() for x_i in x] )
            b = torch.stack( [-e(x_i).double() for x_i in x] )
        else:
            A = torch.Tensor([])
            b = torch.Tensor([])

        Q = torch.stack( [hess_L(x[i],lam[i],c[i] ).double()    for i in range(len(x))] )
        p = torch.stack( [grad_f(x_i,c[i]).double() for i,x_i in enumerate(x)] )

        primaldual = QPFunction_Plus(verbose=-1)(Q,p,G,h,A,b)   #The Hessian is H=2Q, the SQP minimizes (1/2)xT H x, Amos puts in (1/2) automatically

        d     = primaldual[ :,:N ]
        lam_d = primaldual[ :, N:N+M_in ]
        nu_d  = primaldual[ :,   N+M_in:N+M_in+M_eq ]
        x = x+alpha*d
        lam = lam + alpha*(lam_d - lam)

    return x, lam_d






# 0713 reworked from SQP_b

def SQP_c(c, N, M_in, M_eq, g, e, grad_g, grad_e, grad_f, hess_L, alpha, max_iter, primal0 = None, dual0 = None, return_dual = False ):

    if primal0==None:
        dual0=None
    if dual0==None:
        primal0=None

    batsize = c.shape[0]

    #x   = torch.ones(  batsize, N ).double()
    #lam = torch.zeros( batsize, M_in ).double()

    #torch.manual_seed(15)
    #x   = 0.45*torch.ones(  batsize, N ).double() + 0.1*torch.rand(  batsize, N ).double()
    if primal0==None:
        x   = 0.5*torch.ones(  batsize, N ).double()
        lam = torch.zeros( batsize, M_in ).double()
    else:
        x   = primal0
        lam = dual0

    # Ax == b
    # Gx <= h
    for _ in range(max_iter):

        #print("x = " )
        #print( x )
        #print("lam = " )
        #print( lam )


        G = torch.stack( [grad_g(x_i).double() for x_i in x] )
        h = torch.stack( [-g(x_i).double() for x_i in x] )#.unsqueeze(1)
        if e != None:
            A = torch.stack( [grad_e(x_i).double() for x_i in x] )
            b = torch.stack( [-e(x_i).double() for x_i in x] )
        else:
            A = torch.Tensor([])
            b = torch.Tensor([])


        Q = torch.stack( [hess_L(x[i],lam[i],c[i] ).double()    for i in range(len(x))] )
        p = torch.stack( [grad_f(x_i,c[i]).double() for i,x_i in enumerate(x)] )

        primaldual = QPFunction_Plus(verbose=-1)(Q,p,G,h,A,b)   #The Hessian is H=2Q, the SQP minimizes (1/2)xT H x, Amos puts in (1/2) automatically

        #d   = primaldual[ :,:N ]
        #lam = primaldual[ :, N:N+M_in ]
        #nu  = primaldual[ :,   N+M_in:N+M_in+M_eq ]
        #x = x+alpha*d

        d     = primaldual[ :,:N ]
        lam_d = primaldual[ :, N:N+M_in ]
        nu_d  = primaldual[ :,   N+M_in:N+M_in+M_eq ]
        x = x+alpha*d


        lam = lam + alpha*(lam_d - lam)


    if return_dual:
        return torch.cat( (x,lam),1 )
    else:
        return x







# Updated from SQP_a to use lagrangian Hessian
def SQP_b(c, N, g, e, grad_g, grad_e, grad_f, hess_L, alpha, max_iter ):

    batsize = c.shape[0]

    M_in = N+1
    M_eq = 1

    # x is a batch
    x = torch.ones(c.shape).double()

    l = [torch.ones(batsize, M_in), torch.ones(batsize, M_eq)]


    # Ax == b
    # Gx <= h
    for _ in range(max_iter):

        G = torch.stack( [grad_g(x_i).double() for x_i in x] )
        h = torch.stack( [-g(x_i).double() for x_i in x] )
        if e != None:
            A = torch.stack( [grad_e(x_i).double() for x_i in x] )
            b = torch.stack( [-e(x_i).double() for x_i in x] )
        else:
            A = torch.Tensor([])
            b = torch.Tensor([])

        Q = torch.stack( [hess_f(x_i,l[0][i][0] ).double()    for i,x_i in enumerate(x)] )
        p = torch.stack( [grad_f(x_i,c[i]).double() for i,x_i in enumerate(x)] )

        #print("Q.shape = ")
        #print(Q.shape )
        #print("p.shape = ")
        #print(p.shape )
        #print("G.shape = ")
        #print(G.shape )
        #print("h.shape = ")
        #print(h.shape )
        #print("A.shape = ")
        #print(A.shape )
        #print("b.shape = ")
        #print(b.shape)

        """
        print("c = " )
        print( c )
        print("Q = " )
        print( Q )
        print("p = " )
        print( p )
        print("G = " )
        print( G )
        print("h = " )
        print( h )
        print("A = " )
        print( A )
        print("b = " )
        print( b )
        """

        d = QPFunction(verbose=-1)(Q,p,G,h,A,b)   #The Hessian is H=2Q, the SQP minimizes (1/2)xT H x, Amos puts in (1/2) automatically

        x = x+alpha*d

        """
        print("iteration {}".format(_))
        print("d = ")
        print( d    )
        print("x = ")
        print( x    )
        """

    return x






# Adapted from SQP_cvx 0621
# Sequential quadratic programming for general convex problems
# Tested only for a linear-constrained problem

# c is the vector of model parameters
def SQP_a(c, N, g, e, grad_g, grad_e, grad_f, hess_f, alpha, max_iter ):

    # x is a batch
    x = torch.ones(c.shape).double()

    # Ax == b
    # Gx <= h
    for _ in range(max_iter):

        G = torch.stack( [grad_g(x_i).double() for x_i in x] )
        h = torch.stack( [-g(x_i).double() for x_i in x] )
        if e != None:
            A = torch.stack( [grad_e(x_i).double() for x_i in x] )
            b = torch.stack( [-e(x_i).double() for x_i in x] )
        else:
            A = torch.Tensor([])
            b = torch.Tensor([])

        Q = torch.stack( [hess_f(x_i).double() for x_i in x] )
        p = torch.stack( [grad_f(x_i,c[i]).double() for i,x_i in enumerate(x)] )

        #print("Q.shape = ")
        #print(Q.shape )
        #print("p.shape = ")
        #print(p.shape )
        #print("G.shape = ")
        #print(G.shape )
        #print("h.shape = ")
        #print(h.shape )
        #print("A.shape = ")
        #print(A.shape )
        #print("b.shape = ")
        #print(b.shape)

        """
        print("c = " )
        print( c )
        print("Q = " )
        print( Q )
        print("p = " )
        print( p )
        print("G = " )
        print( G )
        print("h = " )
        print( h )
        print("A = " )
        print( A )
        print("b = " )
        print( b )
        """

        d = QPFunction(verbose=-1)(Q,p,G,h,A,b)   #The Hessian is H=2Q, the SQP minimizes (1/2)xT H x, Amos puts in (1/2) automatically

        x = x+alpha*d

        """
        print("iteration {}".format(_))
        print("d = ")
        print( d    )
        print("x = ")
        print( x    )
        """

    return x








# Need oracles for Gradient of f,h,g and Hessian of f
# min f(x)
# st  e_i(x)  = 0  Ai
#     g_j(x) <= 0  Aj
#
# SQP stepsize - alpha

# grad_g should return a 2 tensor G_ij = partial_j (g_i(x))   (Jacobian)
# same for e

#def SQP(f,g,h,grad_f,grad_g,grad_h,hess_f):
# first version - assume
def SQP( Q,q, g,e, grad_g,grad_e, alpha, max_iter ):

    x = torch.zeros(q.shape)

    x = x.double()
    q = q.double()
    Q = Q.double()


    # Ax == b
    # Gx <= h
    for _ in range(max_iter):
        # gradient of f depends on x?
        p = q + torch.bmm( torch.stack([Q for _ in range(len(x))]), x.unsqueeze(1).permute(0,2,1) ).squeeze()
        G = torch.stack( [grad_g(x_i) for x_i in x]  ).double()
        A = torch.stack( [grad_e(x_i) for x_i in x]  ).double()
        h = torch.stack( [-g(x_i) for x_i in x]  ).double()
        b = torch.stack( [-e(x_i) for x_i in x]  ).double()

        d = QPFunction(verbose=-1)(2*Q,p,G,h,A,b)   #The Hessian is H=2Q, the SQP minimizes (1/2)xT H x, Amos puts in (1/2) automatically

        x = x+alpha*d


    return x



# From Boyd's manuscript 'ADMM for distributed...'
def construct_qp_kkt(P,A,b,rho):
    return torch.cat(    (   torch.cat(  ( P+rho*torch.eye(P.shape[0]), A.T ),       1  ),
                             torch.cat(  ( A, torch.zeros(A.shape[0],A.shape[0]) ), 1  )   ),  0   )





def slack_form(P,q,A,b,G,h):
    nineq = G.shape[0]
    n     = G.shape[1]
    neq   = A.shape[0]

    P_ = torch.cat(  (torch.cat( (P, torch.zeros(n,nineq)),1 ),
                      torch.zeros(nineq,n+nineq)) , 0         )
    G_ = torch.cat( (G, torch.eye(nineq)), 1)
    A_ = torch.cat( (A, torch.zeros(neq,nineq)), 1)
    q_ = torch.cat( (q,torch.zeros(nineq)) )
    b_ = b
    h_ = h

    A_ = torch.cat( (G_,A_),0 )
    b_ = torch.cat( (h_,b_) )

    return (P_,q_,A_,b_)  # new asssumption is that x,s >=0 (standard form)


# adapted from slack_form which has x>=0 assumption
# now has no assumption - introduce x+  -  x-  =  x
# Ax==b becomes [A -A 0]z = b
# Gx<=h becomes [G -G I]z = h
# qx    becomes [q -q 0]z
# xQx   becomes  [ Q  -Q  0]
#               z[-Q   Q  0]z
#                [ 0   0  0]
def slack_form_pos(P,q,A,b,G,h):
    nineq = G.shape[0]
    n     = G.shape[1]
    neq   = A.shape[0]   # note neq can be zero if A is empty; use numel()


    PPPP = torch.cat(   (torch.cat( (P,-P), 1 ),
                         torch.cat( (-P,P), 1 )), 0  )
    #P_ = torch.cat(  (torch.cat( (P, torch.zeros(n,nineq)),1 ),
    #                  torch.zeros(nineq,n+nineq)) , 0         )
    P_ = torch.cat(  (torch.cat( (PPPP, torch.zeros(2*n,nineq)),1 ),
                      torch.zeros(nineq,2*n+nineq)) , 0         )
    G_ = torch.cat( (G,-G, torch.eye(nineq)), 1)
    if A.numel() > 0:
        A_ = torch.cat( (A,-A, torch.zeros(neq,nineq)), 1)
    else:
        A_ = A
    # else do nothing
    q_ = torch.cat( (q,-q, torch.zeros(nineq)) )
    b_ = b
    h_ = h

    #print("A_.shape = ")
    #print( A_.shape )
    #print("A_ = ")
    #print( A_ )

    if A.numel() > 0:
        A_ = torch.cat( (G_,A_),0 )
        b_ = torch.cat( (h_,b_) )
    else:
        A_ = G_
        b_ = h_

    return (P_,q_,A_,b_)  # new asssumption is that x,s >=0 (standard form)




# ADMM for general-form QP
# Ax=b, Gx<=h
# All tensor inputs are batch
# sol0: initial solution
# if return_std_form: keep x in terms of [x+, x-, s]
def ADMM_qp(Q,p,G,h,A,b,rho,max_iter, sol0=None, return_std_form = False):

    # Is this assumption safe?
    N = Q[0].shape[0]

    Q_ = []
    p_ = []
    A_ = []
    b_ = []
    for i in range(len(p)):
        # A[i] might not exist (no eq constraint)
        A_i = A[i] if len(A)>0 else torch.Tensor([[]])
        b_i = b[i] if len(b)>0 else torch.Tensor( [] )

        Q_t,p_t,A_t,b_t = slack_form_pos(Q[i],p[i],A_i,b_i,G[i],h[i])

        Q_.append(Q_t)
        p_.append(p_t)
        A_.append(A_t)
        b_.append(b_t)

    Q_ = torch.stack(Q_)
    p_ = torch.stack(p_)
    A_ = torch.stack(A_)
    b_ = torch.stack(b_)

    x_admm = batch_ADMM_QP(Q_,p_,A_,b_,rho,max_iter, xuz0=sol0, return_uz = True)
    #print("x_admm = ")
    #print( x_admm    )
    #print("x_admm.shape = ")
    #print( x_admm.shape    )
    #input("waiting")

    if not return_std_form:
        x_admm = x_admm[:,:N] - x_admm[:,N:2*N]

    return x_admm



# Adapted from ADMM_QP
# now each P,q,A,b is a batch
# rather than q only
# xuz0: initial solution iterate
def batch_ADMM_QP(P,q,A,b,rho,max_iter, xuz0=None, return_uz = False):

    n = q.shape[1]

    #x = torch.zeros(q.shape)
    #u = torch.zeros(q.shape)
    #z = torch.zeros(q.shape)
    if xuz0 == None:
        x = torch.ones(q.shape)
        u = torch.zeros(q.shape)
        z = torch.zeros(q.shape)
        #x = torch.cat(   (   torch.ones(q.shape[0],q.shape[1]//2), #/ b[0],
        #                     torch.ones(q.shape[0],q.shape[1])             ), 1  )
    else:
        x = xuz0[:,:n]
        u = xuz0[:, n:2*n]
        z = xuz0[:,   2*n:3*n]


    x = x.double()
    u = u.double()
    z = z.double()

    bat_size = x.shape[0]   # shape of x (y,z) defines the batch size

    KKT = torch.stack(  [construct_qp_kkt(P[i],A[i],b[i],rho) for i in range(bat_size)]  )

    for _ in range(max_iter):

        rhs = torch.cat(   ( -q + rho*( z - u ),
                              b ), 1  ).unsqueeze(1).permute(0,2,1)
                              #b.repeat(q.shape[0],1) ), 1  ).unsqueeze(1).permute(0,2,1)   # check the cat dimension

        LHS = KKT
        #LHS = KKT.repeat(bat_size,1,1)

        #print("LHS = ")
        #print( LHS    )
        #print("rhs = ")
        #print( rhs    )


        # depending on version
        # X = torch.solve(B, A).solution   other return is L,U
        #  <=>
        # X = torch.linalg.solve(A, B)
        # https://pytorch.org/docs/stable/generated/torch.linalg.solve.html#torch.linalg.solve
        solve_out = torch.linalg.solve(LHS.double(), rhs.double())     #extra arguments for batch dimensions:  , *, out=None)
        #solve_out = torch.solve(rhs, LHS).solution

        #print("solve_out = ")
        #print( solve_out    )
        #print("solve_out.squeeze(2) = ")
        #print( solve_out.squeeze(2)    )

        #x = solve_out.squeeze(2)[:,:P.shape[0]]
        x = solve_out.squeeze(2)[:,:P[0].shape[0]]   # P[0] because P is a batch now
        z = nn.ReLU()( x + u )
        u = u + x - z


    if return_uz:
        return torch.cat( (x,u,z), 1 )
    else:
        return x



# superceded batch_ADMM_QP
# q must be batch
# P, A are 2D
# b is 1D
# QP slack form assumed in Ax=b
def ADMM_QP(P,q,A,b,rho,max_iter):

    #x = torch.zeros(q.shape)
    #u = torch.zeros(q.shape)
    #z = torch.zeros(q.shape)
    x = torch.cat(   (   torch.ones(q.shape[0],q.shape[1]//2) / b[0],
                         torch.ones(q.shape[0],q.shape[1])             ), 1  )
    u = torch.zeros(q.shape)
    z = torch.zeros(q.shape)

    x = x.double()
    u = u.double()
    z = z.double()

    KKT = construct_qp_kkt(P,A,b,rho)

    bat_size = x.shape[0]   # shape of x (y,z) defines the batch size

    for _ in range(max_iter):

        rhs = torch.cat(   ( -q + rho*( z - u ),
                              b.repeat(q.shape[0],1) ), 1  ).unsqueeze(1).permute(0,2,1)   # check the cat dimension

        LHS = KKT.repeat(bat_size,1,1)

        # depending on version
        # X = torch.solve(B, A).solution   other return is L,U
        #  <=>
        # X = torch.linalg.solve(A, B)
        # https://pytorch.org/docs/stable/generated/torch.linalg.solve.html#torch.linalg.solve
        solve_out = torch.linalg.solve(LHS.double(), rhs.double())     #extra arguments for batch dimensions:  , *, out=None)
        #solve_out = torch.solve(rhs, LHS).solution

        x = solve_out.squeeze(2)[:,:P.shape[0]]
        z = nn.ReLU()( x + u )
        u = u + x - z

        # x = solve (z,u)
        # z = relu(x+u)
        # u = u + x - z
    return x


# Unrolled OSQP ADMM solver for QP
# Final form - same functionality as QPTH
# Input Q,c,A,b,G,h
# All batch
# Issues with other qo routines ADMM_QP, ADMM_qp, batch_ADMM_qp:
#   They don't take constraint matrices as batch input
#def OSQP_ADMM(Q,p,G,h,A,b,rho,max_iter, x0=None):







# UNTESTED
# Each input should be batch
# f_opt is 1D
# f_x   is 1D
# grad_f_x is 2D
def polyak_step(f_opt,f_x,grad_f_x):
    return (f_x - f_opt) / (grad_f_x**2).sum(1)



# n = number of items
# C = capacity
# eps = quad reg strength
# rho = augmented lagrangian param + ADMM stepsize
def knapsack_QP_std_form(n,C,eps,rho):


    # Empty Tensor
    e = torch.Tensor()

    # Vector for Fairness constraint LHS and RHS
    knap_vector = torch.ones(1,n)

    # Capacity constrant LHS and RHS
    Clhs       = torch.ones(1,n)
    Crhs       = torch.ones(1) * C

    Clhs_slack = torch.zeros(1,n)

    Blhs =  torch.cat((torch.eye(n,n), torch.eye(n,n)),1 )
    Brhs = torch.ones(n)

    A = torch.cat(   (torch.cat(  (Clhs,Clhs_slack), 1  ),
                      Blhs), 0    )
    b = torch.cat(  (Crhs, Brhs)  )

    # G,h rep 0<=x (for amos QP solver only)
    # x<=1 is reflected in A
    G = Variable( -torch.eye(2*n,2*n) )
    h = Variable(  torch.zeros(2*n)   )

    #  [ x s ]  [ I 0 ]  [x]
    #           [ 0 I ]  [s]
    #
    Q = torch.cat(  (   torch.cat(  ( eps*torch.eye(n),         torch.zeros(n,n) ), 1  ),
                        torch.cat(  (     torch.zeros(n,n), eps*torch.eye(n)     ), 1  )      ), 0  )

    return Q,A,b,G,h






###################################
##### WRAPPERS
###################################

def BlankSQPWrapper( N, M_in, M_eq, g, e, grad_g, grad_e, grad_f, hess_L, alpha, max_iter ):

    class BlankSQP(torch.autograd.Function):

        @staticmethod
        def forward(ctx, c):
            with torch.no_grad():
                x,lam   = SQP_c_dual(c.double(), N, M_in, M_eq, g, e, grad_g, grad_e, grad_f, hess_L, alpha, max_iter )
            return torch.cat( (x,lam),1 ).float()

        @staticmethod
        def backward(ctx, grad_output):
            grad_input   = torch.zeros(grad_output.shape[0],N)
            print("blank SQP bwd function")
            return grad_input

    return BlankSQP.apply



# Wrapper for a function the works on 1D tensors
# Converts the function to work on batches of 1D tensor (2D tensors)
# Input of batch mode function:
#   should be 2D torch tensor
# Output:
#   should be 2D torch tensor
class BatchModeWrapper(nn.Module):
    def __init__(self, f):
        super(BatchModeWrapper, self).__init__()
        self.f = f

    def apply(self,c):
        if len(c.shape) <= 1:
            input("Warning: BatchModeWrapper applied to 1D array")

        out_list = []
        for z in c:
            out_list.append(self.f(z))
        out = torch.stack( out_list )

        return out






# Defunct
# Doesn't work because to carry grad it needs to connect to a Parameter
def BlankIdentityWrapper( N):

    class BlankIdentity(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = torch.zeros(grad_output.shape[0],N)
            print("blank bwd function")
            return grad_input

    return BlankIdentity.apply









# legacy version, new one is in fixedpt_ops.py
def BlankFunctionWrapper( N, f):

    class BlankFn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, c):
            with torch.no_grad():
                #x = f(c.double())
                x = f(c)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = torch.zeros(grad_output.shape[0],N)
            print("blank bwd function")
            return grad_input

    return BlankFn.apply




# cvxpy forward pass with zero backward pass
# takes a cvxpy layer as input
# for now, no way to get the dual solution from that
def BlankCVXWrapper( N, cvxlayer):

    class BlankCVX(torch.autograd.Function):

        @staticmethod
        def forward(ctx, c):
            with torch.no_grad():
                x = cvxlayer(c.double())[0]#.float()
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = torch.zeros(grad_output.shape[0],N)
            print("blank CVX bwd function")
            return grad_input

    return BlankCVX.apply













""" saved before modifying 12/10



# Adapted from ADMM_QP
# now each P,q,A,b is a batch
# rather than q only
def batch_ADMM_QP(P,q,A,b,rho,max_iter, x0 = None):

    #x = torch.zeros(q.shape)
    #u = torch.zeros(q.shape)
    #z = torch.zeros(q.shape)
    if x0 == None:
        x = torch.cat(   (   torch.ones(q.shape[0],q.shape[1]//2), #/ b[0],
                             torch.ones(q.shape[0],q.shape[1])             ), 1  )
    else:
        x = x0

    u = torch.zeros(q.shape)
    z = torch.zeros(q.shape)

    x = x.double()
    u = u.double()
    z = z.double()

    bat_size = x.shape[0]   # shape of x (y,z) defines the batch size

    KKT = torch.stack(  [construct_qp_kkt(P[i],A[i],b[i],rho) for i in range(bat_size)]  )


    for _ in range(max_iter):

        rhs = torch.cat(   ( -q + rho*( z - u ),
                              b ), 1  ).unsqueeze(1).permute(0,2,1)
                              #b.repeat(q.shape[0],1) ), 1  ).unsqueeze(1).permute(0,2,1)   # check the cat dimension

        LHS = KKT
        #LHS = KKT.repeat(bat_size,1,1)

        #print("LHS = ")
        #print( LHS    )
        #print("rhs = ")
        #print( rhs    )


        # depending on version
        # X = torch.solve(B, A).solution   other return is L,U
        #  <=>
        # X = torch.linalg.solve(A, B)
        # https://pytorch.org/docs/stable/generated/torch.linalg.solve.html#torch.linalg.solve
        solve_out = torch.linalg.solve(LHS.double(), rhs.double())     #extra arguments for batch dimensions:  , *, out=None)
        #solve_out = torch.solve(rhs, LHS).solution

        #print("solve_out = ")
        #print( solve_out    )
        #print("solve_out.squeeze(2) = ")
        #print( solve_out.squeeze(2)    )

        #x = solve_out.squeeze(2)[:,:P.shape[0]]
        x = solve_out.squeeze(2)[:,:P[0].shape[0]]   # P[0] because P is a batch now
        z = nn.ReLU()( x + u )
        u = u + x - z

    return x



"""
