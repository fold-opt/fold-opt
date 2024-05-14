import helper_acopf
from helper_acopf import PDLDataSet
import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sqp_functions
from problem import problem, reconstruct_dict
from casadi_solver import casadi_solver, dict_conv, to_cost_dict
from network import cost_coeffs, M
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.1)
args = parser.parse_args()
seed = args.seed
batchsize = args.batchsize
lr = args.lr
torch.manual_seed(seed)

####################################
#       Problem Construction       #
####################################

device = torch.device("cpu")

data_args = {
    'probtype' : 'acopf57',
}

current_path = Path(__file__).absolute().parent
acopf_problem, args = helper_acopf.load_acopf_data(data_args, current_path, device)


####################################
#           Data Loading           #
####################################

load_args = {
    'batchsize' : 1,
    'nworkers' : 0,
    'hiddenfrac' : 0.5,
    'nlayer': 12, 
    'regressorNet_learning_rate' : lr,
    'regressorNet_optimizer': 1,
    'epochs': 10,
    'costdim' : acopf_problem.ngen
}


torch.set_default_dtype(torch.float64)

feature_train_dataset, feature_valid_dataset, feature_test_dataset =  acopf_problem.trainX["feat"][:int(acopf_problem.trainX["feat"].shape[0] / 2), :], acopf_problem.validX["feat"][:int(acopf_problem.validX["feat"].shape[0] / 2), :], acopf_problem.testX["feat"][:int(acopf_problem.testX["feat"].shape[0] / 2), :]

# Training Set #
train_data = np.array(feature_train_dataset)  # , dtype=np.float32)
a = acopf_problem.trainX["pd"][:int(acopf_problem.trainX["pd"].shape[0] / 2), :]
b = acopf_problem.trainX["qd"][:int(acopf_problem.trainX["qd"].shape[0] / 2), :]
c = acopf_problem.trainX["pd_bus"][:int(acopf_problem.trainX["pd_bus"].shape[0] / 2), :]
d = acopf_problem.trainX["qd_bus"][:int(acopf_problem.trainX["qd_bus"].shape[0] / 2), :]
tmp = {"feat":acopf_problem.trainX["feat"][:int(acopf_problem.trainX["feat"].shape[0] / 2), :], "pd":a, "qd":b,
        "pd_bus":c, "qd_bus":d}

parameter_regressor_train_dataset = M(train_data, tmp)

# Validation Set #
valid_data = np.array(feature_valid_dataset)
a = acopf_problem.validX["pd"][:int(acopf_problem.validX["pd"].shape[0] / 2), :]
b = acopf_problem.validX["qd"][:int(acopf_problem.validX["qd"].shape[0] / 2), :]
c = acopf_problem.validX["pd_bus"][:int(acopf_problem.validX["pd_bus"].shape[0] / 2), :]
d = acopf_problem.validX["qd_bus"][:int(acopf_problem.validX["qd_bus"].shape[0] / 2), :]
tmp = {"feat": acopf_problem.validX["feat"][:int(acopf_problem.validX["feat"].shape[0] / 2), :], "pd": a,
        "qd": b, "pd_bus": c, "qd_bus": d}

parameter_regressor_valid_dataset = M(valid_data, tmp)

# Testing Set #
test_data = np.array(feature_test_dataset)
a = acopf_problem.testX["pd"][:int(acopf_problem.testX["pd"].shape[0] / 2), :]
b = acopf_problem.testX["qd"][:int(acopf_problem.testX["qd"].shape[0] / 2), :]
c = acopf_problem.testX["pd_bus"][:int(acopf_problem.testX["pd_bus"].shape[0] / 2), :]
d = acopf_problem.testX["qd_bus"][:int(acopf_problem.testX["qd_bus"].shape[0] / 2), :]
tmp = {"feat": acopf_problem.testX["feat"][:int(acopf_problem.testX["feat"].shape[0] / 2), :], "pd": a,
        "qd": b, "pd_bus": c, "qd_bus": d}

parameter_regressor_test_dataset = M(test_data, tmp)

# Data Loaders #
regressor_train_loader = DataLoader(parameter_regressor_train_dataset, batch_size=load_args['batchsize'], shuffle=False,
                            num_workers=load_args['nworkers'])
regressor_valid_loader = DataLoader(parameter_regressor_valid_dataset, batch_size=load_args['batchsize'], shuffle=False,
                            num_workers=load_args['nworkers'])
regressor_test_loader = DataLoader(parameter_regressor_test_dataset, batch_size=len(parameter_regressor_test_dataset), shuffle=False,
                            num_workers=load_args['nworkers'])


print("Defining regressor model . . .")
prediction_network = cost_coeffs().double().to(device)


if load_args['regressorNet_optimizer'] == 1:
    popt = optim.Adam(prediction_network.parameters(), lr=load_args['regressorNet_learning_rate'])
else:
    popt = optim.SGD(prediction_network.parameters(), lr=load_args['regressorNet_learning_rate'])


loss_fn = torch.nn.L1Loss()

for name, param in prediction_network.named_parameters():
    param.requires_grad = True



###################################
#         Neural Network          #
###################################

epochs = 25

training_precomp = []
validation_precomp = []

solver = casadi_solver(acopf_problem, batchsize)

N =  208
alpha = 0.01

sqp_layer = sqp_functions.LayerSQP(N, 1, acopf_problem, stepsize=alpha)
problem_fn = lambda form, c: problem(form[0], form[1], form[2], form[3], acopf_problem, form[4], form[5], sqp_layer.M_eq, sqp_layer.M_in, c)

# Epochs #
for k in range(epochs):

    # Iterations #
    idx = 0
    for x_in in regressor_train_loader:

        x_feat = x_in[1]["feat"].to(device)
        delta_t = abs(x_feat.flatten()[0] - x_feat.flatten()[1]).detach().cpu().numpy()

        cost = {
                'quad_cost' : acopf_problem.quad_cost*(1 + delta_t / 100),
                'lin_cost' : acopf_problem.lin_cost*(1 + delta_t / 100),
                'const_cost' : acopf_problem.const_cost
            }
    

        if k == 0:
            x, primal_opt, _, _ = solver.solve(acopf_problem.trainX, acopf_problem.trainY, cost, full=True)
            training_precomp.append((x, primal_opt))
        else:
            x = training_precomp[idx][0]
            primal_opt = training_precomp[idx][1]
       
        c_hat = prediction_network(x_feat.flatten())
        
        c_hat_numpy = {}
        c_tensor = []
        c = []
        for elem in c_hat.keys(): 
            c_hat[elem].retain_grad()
            c_hat_numpy[elem] = c_hat[elem].detach().cpu().numpy()
            c_tensor.append(torch.from_numpy(cost[elem]))
            c.append(c_hat[elem])


        c_tensor = torch.stack(c_tensor).to(device)
        c = torch.stack(c).to(device)


        ###################################
        #       Fixed Point SQP           #
        ###################################

        form_in = dict_conv(acopf_problem.trainX, acopf_problem.trainY)
        form = problem_fn(form_in,c_hat)
        x_sqp = sqp_layer.solve(torch.stack([c_hat['quad_cost'], c_hat['lin_cost'], c_hat['const_cost']]).flatten().unsqueeze(0), (c.to(device),form))


        ###################################
        #            Output               #
        ###################################


        reconst_x_true = reconstruct_dict(x, acopf_problem.trainX, form.xindices)
        reconst_x_pred = reconstruct_dict(x, acopf_problem.trainX, form.xindices)
        reconst_y = reconstruct_dict(primal_opt, acopf_problem.trainY, form.yindices)
        reconst_sqp = reconstruct_dict(x_sqp, acopf_problem.trainY, form.yindices)

        cost['quad_cost'] = torch.tensor(cost['quad_cost'])
        cost['lin_cost'] = torch.tensor(cost['lin_cost'])
        cost['const_cost'] = torch.tensor(cost['const_cost'])

        loss = loss_fn(acopf_problem.opt_gap_alt(reconst_x_true, x_sqp, form.yindices, cost), acopf_problem.opt_gap_alt(reconst_x_true, primal_opt, form.yindices, cost))

        popt.zero_grad()
        loss.backward(retain_graph=True)
        popt.step()

        loss_val = loss.item()

        print("\n----- Epoch : ", k, "- Iteration : ", idx, " -----")
        print("Regret                    :  ", loss_val)

        for keys in reconst_x_true.keys():
            reconst_x_true[keys] = reconst_x_true[keys].clone().detach().cpu().numpy()

        for keys in reconst_y.keys():
            reconst_y[keys] = reconst_y[keys].clone().detach().cpu().numpy()

        for keys in reconst_sqp.keys():
            reconst_sqp[keys] = reconst_sqp[keys].clone().detach().cpu().numpy()

        for keys in reconst_x_pred.keys():
            reconst_x_pred[keys] = reconst_x_pred[keys].clone().detach().cpu().numpy()


        eq_viol   = torch.norm(torch.tensor(np.array(acopf_problem.eq_resid(reconst_x_true, reconst_sqp))), p=1).item()
        ineq_viol = torch.norm(torch.tensor(np.array(acopf_problem.ineq_dist(reconst_x_true, reconst_sqp))), p=1).item()
        eq_viol_r   = torch.norm(torch.tensor(np.array(acopf_problem.eq_resid(reconst_x_pred, reconst_sqp))), p=1).item()
        ineq_viol_r = torch.norm(torch.tensor(np.array(acopf_problem.ineq_dist(reconst_x_pred, reconst_sqp))), p=1).item()
        x_hat_obj = acopf_problem.obj_fn(reconst_x_true, reconst_sqp)

        print("Objective (x_hat)         :  ", x_hat_obj)
        print("Objective (x_true)        :  ", acopf_problem.obj_fn(reconst_x_true, reconst_y))
        print("Equality (relative)       :  ", eq_viol_r)
        print("Inequality (relative)     :  ", ineq_viol_r)
        print("--------------------------------------------\n")


        # Save results #
        
        f = open(Path(current_path)/f'log/training_loss_{seed}.txt', 'a')
        f.write(str(loss_val) +'\n')
        f.close()

        f = open(Path(current_path)/f'log/training_obj_{seed}.txt', 'a')
        f.write(str(x_hat_obj) +'\n')
        f.close()

        f = open(Path(current_path)/f'log/training_eq_{seed}.txt', 'a')
        f.write(str(eq_viol_r) +'\n')
        f.close()

        f = open(Path(current_path)/f'log/training_ineq_{seed}.txt', 'a')
        f.write(str(ineq_viol_r) +'\n')
        f.close()

        f = open(Path(current_path)/f'log/training_cdiff_{seed}.txt', 'a')
        f.write(str(torch.norm(c-c_tensor, p=1).item()) +'\n')
        f.close()

        idx +=1



    idx = 0
    for x_in in regressor_valid_loader:

        x_feat = x_in[1]["feat"].to(device)
        delta_t = abs(x_feat.flatten()[0] - x_feat.flatten()[1]).detach().cpu().numpy()


        cost = {
                'quad_cost' : acopf_problem.quad_cost*(1 + delta_t / 100),
                'lin_cost' : acopf_problem.lin_cost*(1 + delta_t / 100),
                'const_cost' : acopf_problem.const_cost
            }

        if k == 0:
            x, primal_opt, _, _ = solver.solve(acopf_problem.trainX, acopf_problem.trainY, cost, full=True)
            training_precomp.append((x, primal_opt))
        else:
            x = training_precomp[idx][0]
            primal_opt = training_precomp[idx][1]
       
        c_hat = prediction_network(x_feat.flatten())

        c_hat_numpy = {}
        c_tensor = []
        c = []
        for elem in c_hat.keys(): 
            c_hat[elem].retain_grad()
            c_hat_numpy[elem] = c_hat[elem].detach().cpu().numpy()
            c_tensor.append(torch.from_numpy(cost[elem]))
            c.append(c_hat[elem])


        c_tensor = torch.stack(c_tensor).to(device)
        c = torch.stack(c).to(device)


        ###################################
        #             Solver              #
        ###################################

        primal, ipopt_form = solver.solve(acopf_problem.trainX, acopf_problem.trainY, c_hat_numpy)
        primal.requires_grad = True
        x_sqp = primal
        

        ###################################
        #            Output               #
        ###################################


        reconst_x_true = reconstruct_dict(x, acopf_problem.trainX, ipopt_form.xindices)
        reconst_x_pred = reconstruct_dict(x, acopf_problem.trainX, ipopt_form.xindices)
        reconst_y = reconstruct_dict(primal_opt, acopf_problem.trainY, ipopt_form.yindices)
        reconst_sqp = reconstruct_dict(x_sqp, acopf_problem.trainY, ipopt_form.yindices)

        cost['quad_cost'] = torch.tensor(cost['quad_cost'])
        cost['lin_cost'] = torch.tensor(cost['lin_cost'])
        cost['const_cost'] = torch.tensor(cost['const_cost'])

        loss = loss_fn(acopf_problem.opt_gap_alt(reconst_x_true, x_sqp, ipopt_form.yindices, cost), acopf_problem.opt_gap_alt(reconst_x_true, primal_opt, ipopt_form.yindices, cost))

        loss_val = loss.item()

        print("\n----- Epoch : ", k, "- Validation : ", idx, " -----")
        print("Regret                    :  ", loss_val)


        for keys in reconst_x_true.keys():
            reconst_x_true[keys] = reconst_x_true[keys].clone().detach().cpu().numpy()

        for keys in reconst_y.keys():
            reconst_y[keys] = reconst_y[keys].clone().detach().cpu().numpy()

        for keys in reconst_sqp.keys():
            reconst_sqp[keys] = reconst_sqp[keys].clone().detach().cpu().numpy()

        for keys in reconst_x_pred.keys():
            reconst_x_pred[keys] = reconst_x_pred[keys].clone().detach().cpu().numpy()


        eq_viol   = torch.norm(torch.tensor(np.array(acopf_problem.eq_resid(reconst_x_true, reconst_sqp))), p=1).item()
        ineq_viol = torch.norm(torch.tensor(np.array(acopf_problem.ineq_dist(reconst_x_true, reconst_sqp))), p=1).item()
        eq_viol_r   = torch.norm(torch.tensor(np.array(acopf_problem.eq_resid(reconst_x_pred, reconst_sqp))), p=1).item()
        ineq_viol_r = torch.norm(torch.tensor(np.array(acopf_problem.ineq_dist(reconst_x_pred, reconst_sqp))), p=1).item()
        x_hat_obj = acopf_problem.obj_fn(reconst_x_true, reconst_sqp)

        print("Objective (x_hat)         :  ", x_hat_obj)
        print("Objective (x_true)        :  ", acopf_problem.obj_fn(reconst_x_true, reconst_y))
        print("Equality (relative)       :  ", eq_viol_r)
        print("Inequality (relative)     :  ", ineq_viol_r)
        print("--------------------------------------------\n")


        # Save results #
        
        f = open(Path(current_path)/f'log/validation_loss_{seed}.txt', 'a')
        f.write(str(loss_val) +'\n')
        f.close()

        f = open(Path(current_path)/f'log/validation_obj_{seed}.txt', 'a')
        f.write(str(x_hat_obj) +'\n')
        f.close()

        f = open(Path(current_path)/f'log/validation_eq_{seed}.txt', 'a')
        f.write(str(eq_viol_r) +'\n')
        f.close()

        f = open(Path(current_path)/f'log/validation_ineq_{seed}.txt', 'a')
        f.write(str(ineq_viol_r) +'\n')
        f.close()

        f = open(Path(current_path)/f'log/validation_cdiff_{seed}.txt', 'a')
        f.write(str(torch.norm(c-c_tensor, p=1).item()) +'\n')
        f.close()

        idx +=1


    ###################################
    #           Recording             #
    ###################################

    if True:

        f = open(Path(current_path)/f'log/training_loss_{seed}.txt', 'a')
        f.write('\n')
        f.close()

        f = open(Path(current_path)/f'log/training_obj_{seed}.txt', 'a')
        f.write('\n')
        f.close()

        f = open(Path(current_path)/f'log/training_eq_{seed}.txt', 'a')
        f.write('\n')
        f.close()

        f = open(Path(current_path)/f'log/training_ineq_{seed}.txt', 'a')
        f.write('\n')
        f.close()

        f = open(Path(current_path)/f'log/training_cdiff_{seed}.txt', 'a')
        f.write('\n')
        f.close()

        f = open(Path(current_path)/f'log/validation_loss_{seed}.txt', 'a')
        f.write('\n')
        f.close()

        f = open(Path(current_path)/f'log/validation_obj_{seed}.txt', 'a')
        f.write('\n')
        f.close()

        f = open(Path(current_path)/f'log/validation_eq_{seed}.txt', 'a')
        f.write('\n')
        f.close()

        f = open(Path(current_path)/f'log/validation_ineq_{seed}.txt', 'a')
        f.write('\n')
        f.close()

        f = open(Path(current_path)/f'log/validation_cdiff_{seed}.txt', 'a')
        f.write('\n')
        f.close()




print("Training complete.")
