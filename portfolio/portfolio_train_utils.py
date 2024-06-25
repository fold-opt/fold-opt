from portfolio_task_solver import get_markowitz_constraints_cvx, solve_markowitz_cvx
import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import sys
sys.path.append('.')
import unrolled_ops

import gurobipy as gp
from gurobipy import GRB


def test_regret(model, test_inputs, test_targets, constraints, variables):
    with torch.no_grad():

        test_outputs = model(test_inputs)

        true_sol = []
        pred_sol = []
        for i in range(len(test_inputs)):
            true = solve_markowitz_cvx(constraints,variables,test_targets[i])
            pred = solve_markowitz_cvx(constraints,variables,test_outputs[i])
            true_sol.append( torch.Tensor(true) )
            pred_sol.append( torch.Tensor(pred) )

        true_sol = torch.stack(true_sol)
        pred_sol = torch.stack(pred_sol)

        batsize = len(test_targets)
        vecsize = len(test_targets[0])

        regret = torch.bmm( test_targets.view(batsize,1,vecsize), (true_sol - pred_sol).view(batsize,vecsize,1) ).squeeze()

    return regret


# c_true are generally the target data
# c_pred are then the model-predicted values
# input and output are expected to be torch tensors
def spo_grad_grb(c_true, c_pred, solver, variables):

    c_spo = (2*c_pred - c_true)
    grad = []
    regret = []
    for i in range(len(c_spo)):    # iterate the batch
        sol_true = solve_markowitz_grb(solver,variables, np.array(c_true[i]))
        sol_spo  = solve_markowitz_grb(solver,variables, np.array(c_spo[i]))
        grad.append(  torch.Tensor(sol_spo - sol_true)  )
        regret.append(  torch.dot(c_true, torch.Tensor(sol_true - sol_pred)  ) )   # this is only for diagnostic / results output

    grad = torch.stack( grad )
    regret = torch.stack( regret )
    return grad, regret

# model is the ML/NN
# optimizer is the torch object
# solver is the CO/LP/QP solver
# variables is the solver's variable handles
def train_fwdbwd_spo_grb(model, optimizer, solver, variables, input_bat, target_bat):

    c_true = target_bat
    c_pred = model(input_bat)
    grad, regret = spo_grad_grb(c_true, c_pred, solver, variables)
    optimizer.zero_grad()
    c_pred.backward(gradient = grad)
    optimizer.step()

    return regret




def spo_grad_cvx(c_true, c_pred, constraints, variables):

    c_spo = (2*c_pred - c_true)
    grad = []
    regret = []
    for i in range(len(c_spo)):    # iterate the batch
        sol_true = solve_markowitz_cvx(constraints,variables, np.array(c_true[i].detach()))
        sol_pred = solve_markowitz_cvx(constraints,variables, np.array(c_pred[i].detach()))
        sol_spo  = solve_markowitz_cvx(constraints,variables, np.array(c_spo[i].detach()))

        grad.append(  torch.Tensor(sol_spo - sol_true)  )
        regret.append(  torch.dot(c_true[i], torch.Tensor(sol_true - sol_pred)  ) )   # this is only for diagnostic / results output

    grad = torch.stack( grad )
    regret = torch.stack( regret )
    return grad, regret



def train_fwdbwd_spo_cvx(model, optimizer, constraints, variables, input_bat, target_bat):

    c_true = target_bat
    c_pred = model(input_bat)
    grad, regret = spo_grad_cvx(c_true, c_pred, constraints, variables)
    optimizer.zero_grad()
    c_pred.backward(gradient = grad)
    optimizer.step()

    return regret



def train_fwdbwd_blackbox_cvx(model, optimizer, blackbox_layer, input_bat, target_bat):

    c_true = target_bat
    c_pred = model(input_bat)
    solver_pred_out = blackbox_layer.apply( c_pred )
    solver_true_out = blackbox_layer.apply( c_true )

    batsize = len(c_pred)
    vecsize = len(c_pred[0])

    # torch.bmm(a.view(7,1,5),b.view(7,5,1))
    # batch dot product
    regret = torch.bmm( c_true.view(batsize,1,vecsize), (solver_true_out - solver_pred_out).view(batsize,vecsize,1) ).squeeze()

    optimizer.zero_grad()
    regret.mean().backward()
    optimizer.step()

    return regret




def train_fwdbwd_cvxpy(model, optimizer, cvxpy_layer, blackbox_layer, input_bat, target_bat):

    c_true = target_bat
    c_pred = model(input_bat)
    solver_pred_out, = cvxpy_layer( c_pred, solver_args={'eps':1e-9} )
    with torch.no_grad():
        solver_true_out  = blackbox_layer.apply( c_true )
        solver_pred_hard = blackbox_layer.apply( c_pred )

    batsize = len(c_pred)
    vecsize = len(c_pred[0])

    # torch.bmm(a.view(7,1,5),b.view(7,5,1))
    # batch dot product
    regret = torch.bmm( c_true.view(batsize,1,vecsize), (solver_true_out - solver_pred_out).view(batsize,vecsize,1) ).squeeze()

    optimizer.zero_grad()
    regret.mean().backward()
    optimizer.step()

    with torch.no_grad():
        regret_out = torch.bmm( c_true.view(batsize,1,vecsize), (solver_true_out - solver_pred_hard).view(batsize,vecsize,1) ).squeeze()

    return regret_out




def BlackboxMarkowitzWrapper(constraints, variables, lambd):

    class BlackboxMarkowitz(torch.autograd.Function):

        @staticmethod
        def forward(ctx, c):
            y = []
            for i in range(len(c)):
                sol = solve_markowitz_cvx(constraints,variables,c[i].detach())  #make sure this is doing the right thing over the batch
                y.append(torch.Tensor(sol))
            y = torch.stack( y )
            ctx.save_for_backward( c,y )
            return y

        @staticmethod
        def backward(ctx, grad_output):
            c,y = ctx.saved_tensors
            c_p =  c +  grad_output * lambd
            y_lambd = []
            for i in range(len(c_p)):
                sol = solve_markowitz_cvx(constraints,variables,c_p[i].detach())
                y_lambd.append( torch.Tensor(sol) )
            y_lambd = torch.stack( y_lambd )
            # multiply each gradient by the jacobian for the corresponding sample
            # then restack the results to preserve the batch gradients' format
            grad_input = - 1/lambd*(  y - y_lambd  )

            return grad_input

    return BlackboxMarkowitz







def get_cvxpy_layer(n,COV,gamma,quadreg):

    x = cp.Variable(n)
    c = cp.Parameter(n)

    constraints = [  2*cp.quad_form( x,COV ) <= gamma,
                     e @ x <= 1, x >= 0 ]

    problem = cp.Problem(cp.Maximize( c @ x  - quadreg*cp.pnorm( x, p=2) ),
                      constraints)

    assert problem.is_dpp()
    cvxlayer = CvxpyLayer(problem, parameters=[c], variables=[x])


    return cvxlayer



def get_sqp_layer(n,COV,gamma,quadreg, alpha=1.0, max_iter=1, return_dual = False):

    # Maximize   c @ x  -  quadreg * x^T I x
    # x^T COV x - gamma   <=   0
    # -x   <=   0
    #  e @ x -1 <= 0
    COV_t = torch.Tensor( COV ).double()
    eps = quadreg


    g       = lambda x: torch.cat( ((x@COV_t@x).unsqueeze(0) - gamma,    x.sum().unsqueeze(0)-1.0,     -x), 0 )
    grad_g  = lambda x: torch.cat( (2.0*(COV_t@x).unsqueeze(0),   torch.ones(len(x)).unsqueeze(0),   -torch.eye(len(x))), 0 )
    # stack the gradient of each component of g sideways

    grad_f  = lambda x,c:    -c + 2.0*eps*x
    hess_L  = lambda x,l,c:    2.0*eps*torch.eye(n)  + 2.0* l[0]* COV_t

    #alpha = 1.0
    #max_iter = 60
    def sqplayer(p, primal0=None, dual0=None):
        SQP_soln = unrolled_ops.SQP_c(p.double(), n, n+2, 0, g, None, grad_g, None, grad_f, hess_L, alpha, max_iter, primal0=primal0, dual0=dual0, return_dual = return_dual).float()
        return SQP_soln

    return sqplayer





def train_fwdbwd_sqp(model, optimizer, sqp_layer, blackbox_layer, input_bat, target_bat):

    c_true = target_bat
    c_pred = model(input_bat)
    solver_pred_out = sqp_layer( c_pred )
    with torch.no_grad():
        solver_true_out  = blackbox_layer.apply( c_true )
        solver_pred_hard = blackbox_layer.apply( c_pred )

    batsize = len(c_pred)
    vecsize = len(c_pred[0])


    # torch.bmm(a.view(7,1,5),b.view(7,5,1))
    # batch dot product
    #obj = torch.bmm( c_true.view(batsize,1,vecsize), solver_pred_out.view(batsize,vecsize,1) ).squeeze()
    obj = -(c_true * solver_pred_out).sum(1)


    optimizer.zero_grad()
    obj.mean().backward()
    optimizer.step()

    with torch.no_grad():
        regret_out = torch.bmm( c_true.view(batsize,1,vecsize), (solver_true_out - solver_pred_hard).view(batsize,vecsize,1) ).squeeze()

    return regret_out


# same as train_fwdbwd_cvxpy but we don't use regret, just maximize objective
# use a blackbox_layer to compute the true/rounded/unregularized solution
def train_fwdbwd_cvxpy_noregret(model, optimizer, cvxpy_layer, blackbox_layer, input_bat, target_bat):

    c_true = target_bat
    c_pred = model(input_bat)
    solver_pred_out, = cvxpy_layer( c_pred, solver_args={'eps':1e-9, 'max_iters':5000} )
    with torch.no_grad():
        solver_true_out  = blackbox_layer.apply( c_true )
        solver_pred_hard = blackbox_layer.apply( c_pred )

    batsize = len(c_pred)
    vecsize = len(c_pred[0])

    # torch.bmm(a.view(7,1,5),b.view(7,5,1))
    # batch dot product
    #obj = torch.bmm( c_true.view(batsize,1,vecsize), solver_pred_out.view(batsize,vecsize,1) ).squeeze()
    obj = -(c_true * solver_pred_out).sum(1)


    optimizer.zero_grad()
    obj.mean().backward()
    optimizer.step()

    with torch.no_grad():
        regret_out = torch.bmm( c_true.view(batsize,1,vecsize), (solver_true_out - solver_pred_hard).view(batsize,vecsize,1) ).squeeze()

    return regret_out




# Takes a single numpy array
class gurobiSolver():
    def __init__(self,n,COV,gamma,quadreg, return_dual=False):
        #super(gurobiSolver, self).__init__()
        e = np.ones(n)
        #COV = np.matmul(L,L.T) + np.eye(n)*(0.01*tau)**2
        #w_ = e/10
        #gamma = 2.25 * np.matmul( np.matmul(w_,COV), w_ ) # no factor needed for quadratic reg

        x = cp.Variable(n)
        c = cp.Parameter(n)

        env = gp.Env(empty=True)
        env.setParam("OutputFlag",0)
        if return_dual:
            #print("set QCPDual")
            env.setParam("QCPDual",1)
        env.start()
        m = gp.Model("portfolio", env=env)


        #x = m.addMVar(shape=n, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
        x = m.addMVar(shape=n,  vtype=GRB.CONTINUOUS, name="x")

        self.quad_constr  = m.addConstr(x@(COV@x) <= gamma, name="risk")
        self.lin_constr   = m.addConstr(e @ x <= 1      , name="sum")
        self.lower_bounds = m.addConstr(0 <= x      , name="lb") ##

        self.m = m
        self.env = env
        self.x = x
        self.quadreg = quadreg
        self.return_dual = return_dual

    # coeffs are assumed numpy
    def solve(self,coeffs):

        self.m.setObjective(coeffs @ self.x  - self.quadreg * (self.x@self.x), GRB.MAXIMIZE)
        self.m.optimize()

        #print("self.quad_constr.QCPi = ")
        #print( self.quad_constr.QCPi    )
        #print("self.lin_constr.Pi = ")
        #print( self.lin_constr.Pi    )
        #print("self.lower_bounds.Pi = ")
        #print( self.lower_bounds.Pi    )


        # append 2 dual values to the primal, 1 for quad constr and 1 for sum constr
        # for some reason, QCPi returns as float and Pi returns as len-1 list; hence Pi[0]
        if self.return_dual:
            #print("self.x.X")
            #print( self.x.X )
            #print("self.quad_constr.QCPi")
            #print( self.quad_constr.QCPi )
            #print("self.lin_constr.Pi")
            #print( self.lin_constr.Pi )
            #print("np.array(self.lower_bounds.Pi)")
            #print( np.array(self.lower_bounds.Pi) )
            #input()
            return np.concatenate(  (self.x.X, np.array([self.quad_constr.QCPi, self.lin_constr.Pi]), np.array(self.lower_bounds.Pi))  )


        return self.x.X   #self.quad_constr.getAttr("QCPi")  #



class gurobiBatchSolver():
    def __init__(self,n,COV,gamma,quadreg, return_dual=False):
        self.solver = gurobiSolver(n,COV,gamma,quadreg, return_dual=return_dual)

    # coeffs are a batch of Torch tensors
    def solve(self,coeffs):
        coeffs = coeffs.detach().numpy()
        outs = []
        for c in coeffs:
            outs.append( torch.Tensor(  self.solver.solve(c)  ) )
        outs = torch.stack(outs).double()
        return outs



class PortfolioDiffSQP():
    def __init__(self,n,COV,gamma,quadreg):
        solver = gurobiBatchSolver(n,COV,gamma,quadreg, return_dual=True)#  get_sqp_layer(n,p,tau,L,quadreg, return_dual = False)
        self.blank_solver = unrolled_ops.BlankFunctionWrapper(n,solver.solve)
        self.sqp_layer = get_sqp_layer(n,COV,gamma,quadreg, return_dual = True)
        self.n = n


    # a batch of pytorch coefficients
    def solve(self,coeffs):

        primaldual_init = self.blank_solver(coeffs)
        primal_init = primaldual_init[:,:self.n ]
        dual_init   = primaldual_init[:, self.n:]

        sol = self.sqp_layer(coeffs, primal0 = primal_init,  dual0 = dual_init)
        print("(sol - primaldual_init).abs().max() = ")
        print( (sol - primaldual_init).abs().max()    )
        return sol
