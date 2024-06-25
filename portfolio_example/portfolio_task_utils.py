from portfolio_task_solver import get_markowitz_constraints_cvx, solve_markowitz_cvx
import numpy as np
import torch


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

        # batch dot product
        regret = torch.bmm( test_targets.view(batsize,1,vecsize), (true_sol - pred_sol).view(batsize,vecsize,1) ).squeeze()

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

    # batch dot product
    regret = torch.bmm( c_true.view(batsize,1,vecsize), (solver_true_out - solver_pred_out).view(batsize,vecsize,1) ).squeeze()

    optimizer.zero_grad()
    regret.mean().backward()
    optimizer.step()

    return regret






def train_fwdbwd_SQP(model, optimizer, SQP_layer, input_bat, target_bat, COV_t, gamma):

    c_true = target_bat
    c_pred = model(input_bat)

    #print("COV_t")
    #print( COV_t )
    #print("torch.sort(c_pred,descending=True) = ")
    #print( torch.sort(c_pred,descending=True)    )
    #print("torch.sort(c_true,descending=True) = ")
    #print( torch.sort(c_true,descending=True)    )
    solver_pred_out = SQP_layer( c_pred )
    solver_true_out = SQP_layer( c_true )

    #print("[(x@(COV_t.float()@x)).item() for x in solver_pred_out]")
    #print( [(x@(COV_t.float()@x)).item() for x in solver_pred_out] )
    #print("gamma")
    #print( gamma )

    #print("torch.sort(solver_pred_out,descending=True)")
    #print( torch.sort(solver_pred_out,descending=True) )
    #input()
    batsize = len(c_pred)
    vecsize = len(c_pred[0])

    # batch dot product
    regret = torch.bmm( c_true.view(batsize,1,vecsize), (solver_true_out - solver_pred_out).view(batsize,vecsize,1) ).squeeze()
    #opt = (c_true * solver_pred_out).sum(1).mean()


    optimizer.zero_grad()
    regret.mean().backward()
    #(-opt).mean().backward()
    optimizer.step()

    return  regret #opt



# TODO: the markowitz solving has to be batchified
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
