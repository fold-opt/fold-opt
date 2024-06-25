import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import torch.autograd.functional as func

from torch.func import jacrev
import fold_opt.GMRES as GMRES


class FoldOptLayer(torch.nn.Module):
    def __init__(self, solver, update_step, n_iter=100, backprop_rule='FPI'):
        super().__init__()

        self.n_iter = n_iter
        self.update_step  = update_step
        self.backprop_rule = backprop_rule
        self.solver_blank = BlankFunctionWrapper(solver)

        if backprop_rule=='FPI':
            self.fixedPtModule = FixedPtDiffLFPI()
        elif backprop_rule=='GMRES':
            self.fixedPtModule = FixedPtDiffGMRES()
        else:
            self.fixedPtModule = FixedPtDiffJacobianx()


    def forward(self, c):

        x_star = self.solver_blank(c)
        x_star_step = self.update_step(c, x_star )
        x_return  = self.fixedPtModule.apply(c, x_star_step, x_star, self.n_iter)
        return x_return




class FixedPtDiffLFPI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, x_star_step, x_star, max_iter):
        ctx.save_for_backward(c, x_star_step, x_star, torch.tensor(max_iter))
        return x_star

    @staticmethod
    def backward(ctx, grad_output):
        c, x_star_step, x_star, max_iter = ctx.saved_tensors
        grad_input = JgP_LFPI(c, x_star_step, x_star, grad_output, n_steps=max_iter.item())
        return grad_input.float(), None, None, None, None, None

class FixedPtDiffGMRES(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, x_star_step, x_star, max_iter):
        ctx.save_for_backward(c, x_star_step, x_star, torch.tensor(max_iter))
        return x_star

    @staticmethod
    def backward(ctx, grad_output):
        c, x_star_step, x_star, max_iter = ctx.saved_tensors
        grad_input = JgP_GMRES(c, x_star_step, x_star, grad_output, n_steps=max_iter.item())
        return grad_input.float(), None, None, None, None, None


class FixedPtDiffJacobian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, x_star_step, x_star, max_iter):
        ctx.save_for_backward(c, x_star_step, x_star, torch.tensor(max_iter))
        return x_star

    @staticmethod
    def backward(ctx, grad_output):
        c, x_star_step, x_star, max_iter = ctx.saved_tensors
        jacobian_x, jacobian_c = jacobian_x_c(c, x_star_step, x_star)
        I = torch.eye(jacobian_x.shape[1]).repeat(jacobian_x.shape[0],1,1).to(x_star.dtype)
        dxdc = torch.linalg.solve(I-jacobian_x.to(x_star.dtype),jacobian_c.to(x_star.dtype))
        grad_input   =  torch.bmm(dxdc,grad_output.to(dxdc.dtype).unsqueeze(2)).squeeze(2)
        return grad_input.float(), None, None, None



class FixedPtDiffJacobianx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, x_star_step, x_star, max_iter):
        ctx.save_for_backward(c, x_star_step, x_star, torch.tensor(max_iter))
        return x_star

    @staticmethod
    def backward(ctx, grad_output):
        c, x_star_step, x_star, max_iter = ctx.saved_tensors
        PHI = jacobian_x(c, x_star_step, x_star)
        I = torch.eye(PHI.shape[1]).repeat(PHI.shape[0],1,1).to(x_star.dtype)
        v = torch.linalg.solve(I-PHI.permute(0,2,1).to(x_star.dtype),grad_output.to(x_star.dtype))  # (I - PHI)^T v = g
        grad_input = torch.autograd.grad(x_star_step, c, v, retain_graph=True)[0].detach()          # g^T J = v^T PSI
        return grad_input.float(), None, None, None


def jacobian_x_c(c, x_star_step, x_star):

    N = x_star.shape[1]
    B = x_star.shape[0]

    I = torch.eye(N)

    jacobian_x = [torch.autograd.grad(x_star_step, x_star, grad_outputs=I[i].repeat(B,1), retain_graph=True)[0] for i in range(N)]
    jacobian_x = torch.stack(jacobian_x).permute(1,0,2)

    jacobian_c = [torch.autograd.grad(x_star_step,    c,   grad_outputs=I[i].repeat(B,1), retain_graph=True)[0] for i in range(N)]
    jacobian_c = torch.stack(jacobian_c).permute(1,0,2)

    return jacobian_x, jacobian_c



def jacobian_x(c, x_star_step, x_star):

    N = x_star.shape[1]
    B = x_star.shape[0]

    I = torch.eye(N)

    jacobian_x = [torch.autograd.grad(x_star_step, x_star, grad_outputs=I[i].repeat(B,1), retain_graph=True)[0] for i in range(N)]
    jacobian_x = torch.stack(jacobian_x).permute(1,0,2)
    return jacobian_x





def JgP_LFPI(c, x_star_step, x_star_blank, g, n_steps = 1000, solver=None):

    N = x_star_blank.shape[1]
    B = x_star_blank.shape[0]

    v = torch.autograd.grad(x_star_step, x_star_blank, g, retain_graph=True)[0].detach()

    for i in range(n_steps):
        v = torch.autograd.grad(x_star_step, x_star_blank, v, retain_graph=True)[0].detach() + g

    J = torch.autograd.grad(x_star_step, c, v, retain_graph=True)[0].detach()

    return J.float()






def JgP_GMRES(c, x_star_step, x_star, g, n_steps = 1000, tol = 1e-8):

    N = x_star.shape[1]
    B = x_star.shape[0]
    M = n_steps
    M = max(M,3)
    M = min(M,N+1)

    x0 = torch.autograd.grad(x_star_step, x_star, g, retain_graph=True)[0].detach()
    b  = g



    H = torch.zeros(M + 1, M).repeat(B,1,1)
    Q = torch.zeros(M,N).repeat(B,1,1)
    O = torch.eye(M).repeat(B,1,1)

    jgp = x0 - torch.autograd.grad(x_star_step, x_star, x0, retain_graph=True)[0].detach()
    r =  b - jgp
    #r =  b - v_matrix_mult(A, x0)
    I2, I2skew, r, beta, Q, e = GMRES.v_setup(r, b, x0, H, Q, O, M)

    for k in range(M-1):
        #y = v_matrix_mult(A, Q[k])
        #y = v_matrix_mult(A, Q[:,k,:])
        y =  Q[:,k,:] - torch.autograd.grad(x_star_step, x_star, Q[:,k,:], retain_graph=True)[0].detach()
        H,Q = GMRES.v_update_H_Q(H, Q, y, k, tol)
        if k==0:
            # Initialize the O and R factors
            O, R = GMRES.v_get_O_R(H, O, I2, I2skew, k)
        else:
            # Create the Givens matrix and use it to update O
            # Recover R from O and H
            O, R = GMRES.v_update_O_R(H, O, I2, I2skew, k)
            # Create the RHS of the least squares problem
            # Least squares is equivalent to this back-substitution
            z = GMRES.v_solve_x(Q,O,R,x0,e,beta,k)



    gradient = torch.autograd.grad(x_star_step, c, z, retain_graph=True)[0].detach()


    return gradient.float()






def Jacobian_GMRES(c, x_star_step, x_star, g, n_steps = 1000, solver=None):

    N = x_star.shape[1]
    B = x_star.shape[0]


    J_x = jacobian_x(c, x_star_step, x_star)
    x0 = torch.autograd.grad(x_star_step, x_star, g, retain_graph=True)[0].detach()
    AT = torch.eye(N).repeat(B,1,1) - J_x.permute(0,2,1)
    z = GMRES.GMRes_mod_vec(AT, g, x0, n_steps, tol=1e-8)
    gradient = torch.autograd.grad(x_star_step, c, z, retain_graph=True)[0].detach()


    return gradient.float()


###################################
#           WRAPPERS              #
###################################

"""
BatchModeWrapper

Wrapper for a function that works on 1D tensors. Converts the function to work on batches of
1D tensor (2D tensors).

Input of batch mode function should be 2D torch tensor
Output should be 2D torch tensor

"""
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



def BlankIdentityWrapper( N):

    class BlankIdentity(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = torch.zeros(grad_output.shape[0],N)
            return grad_input

    return BlankIdentity.apply




def BlankFunctionWrapper(f):

    class BlankFn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, c):
            N = c.shape[1]
            ctx.save_for_backward(torch.tensor(N))
            with torch.no_grad():
                x = f(c)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            N, = ctx.saved_tensors
            grad_input = torch.zeros(grad_output.shape[0],N.item())
            return grad_input

    return BlankFn.apply
