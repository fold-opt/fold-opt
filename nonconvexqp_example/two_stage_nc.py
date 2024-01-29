'''

TODO:

v Given the optimization problem (portfolio, acopf) load the corresponding saved model (it can be a dc3, lagrangian dual or primal-dual model)

    portfolio | dc3 ld pdl
    acopf57   |     ld pdl
    knapsack  |     ld pdl

v Load the problem parameters (c) dataset

v Define a feature generator network N

v Generate the feature z dataset by feeding N with c : c -> FGN -> z

v Define a prediction network M

v z -> M -> \hat{c} -> mse(c,hat{c})

v Train by minimizing mse(c,hat{c})

v Evaluate regret : f(x^*(\hat{c)),c) - f(x^*(c),c)

- Compute solution error

'''
import copy
import json
import torch.nn.functional as F
import torch
import numpy as np
from helper_acopf import NNPrimalACOPFSolver, load_acopf_data, load_acopf_predopt_data, NNPredoptACOPFSolver, ACOPFProblem, ACOPFPredopt # JK
#from helper_new_portfolio import NNPrimalSolver, load_portfolio_data, PortfolioProblem, solve_convexqp
#from helper_knapsack import load_knapsack_data, KnapsackProblem, solve_knapsack
from helper_qcqp import NNPrimalSolverQCQP, NNDualSolverQCQP, QCQPDataSet, QCQPSelfSupervisedDataSet, load_qcqp_data, load_qcqp_predopt_data, NNPredoptQCQPSolver, featureNetQCQP, solve_qcqp, QCQPProblem#, featureNet #VDVF
import time, argparse
from setproctitle import setproctitle
from utils import set_seed, dict_agg
import default_args
from pprint import pprint
from utils import str_to_bool, dict_agg, set_seed
#from baseline_dc3 import NNSolver
import torch.nn as nn
import operator
from functools import reduce
import torch.optim as optim
from pathlib import Path
from dataset import Dataset as D
from helper_acopf import acopf_featureNet
from sklearn.preprocessing import normalize
from numpy import linalg as LA
import pandas as panda
from helper_qp_new import NNPrimalSolver, NNDualSolver, load_qp_data, solve_nonconvexqp, NonconvexQPProblem
#import gurobipy as gp
#from gurobipy import GRB
import helper_bilinear
from helper_bilinear import BilinearQPProblem, load_bilinear_data, BiLinearSolver#, bilinear_solver
from pdl import PDLDataSet
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE", DEVICE, flush=True)


#from baseline_supervised import BaselineDataSet

CURRENT_PATH = Path(__file__).absolute().parent

class predNet(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        layer_sizes = [input_dim]
        if args['nlayer']>1:
            layer_sizes += (args['nlayer']-1)*[args['hiddensize']]
            #layers = reduce(operator.add, [[nn.Linear(a,b), nn.ReLU()] for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
            layers = reduce(operator.add, [[nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.1)] for a, b in
                                           zip(layer_sizes[0:-1], layer_sizes[1:])])
            layers += [nn.Linear(layer_sizes[-1], output_dim)] #, nn.Hardsigmoid()]
        else:
            layers = [nn.Linear(layer_sizes[-1], output_dim)]

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight) ## TRY SOMETHING DIFFERENT
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(DEVICE)
        return self.net(x)

class featureNet(nn.Module):
    def __init__(self, nlayer, hidden_size, data):
        super().__init__()
        self._data = data
        if nlayer>1:
            layer_sizes = [data.trainX.shape[1]]
            layer_sizes += (nlayer-1) * [hidden_size] #self._args['featNet_nlayer'] * [self._args['featNet_hiddensize']]
            layers = reduce(operator.add,[[nn.Linear(a, b), nn.ReLU()] for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
            feat_dim = data.trainX.shape[1]
            layers += [nn.Linear(layer_sizes[-1], feat_dim)]
        else:
            feat_dim = data.trainX.shape[1]
            layers = [nn.Linear(data.trainX.shape[1], feat_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out

def feature_Generation(feature_Net, train_data, valid_data, test_data):
    return feature_Net(train_data.to(DEVICE)), feature_Net(valid_data.to(DEVICE)), feature_Net(test_data.to(DEVICE))

def TVT(prediction_Net, train_loader, valid_loader, test_loader, args, problem_class):

    popt = optim.Adam(prediction_Net.parameters(), lr=args['lr']*50)
    scheduler = torch.optim.lr_scheduler.StepLR(popt, step_size=1, gamma=0.99)
    loss = torch.nn.MSELoss()
    patience = 200
    best = 10000
    no_good = 0
    best_model = copy.deepcopy(prediction_Net)
    to_evaluate = True

    if 'nonconvexqp' in args['probtype'] or 'bilinear' in args['probtype']: # and to_evaluate == False: # and 1==0:
        for e in range(30):
            t0 = time.time()
            train_total_loss = 0.
            prediction_Net.train()
            print("Entering the training routine . . .")
            print("Epoch: ", e+1)
            for i, data in enumerate(train_loader):
                z, c = data['X'], data['Y']
                z, c = z.to(DEVICE), c.to(DEVICE)
                popt.zero_grad()
                c_hat = prediction_Net(z)
                train_loss = loss(c_hat, c) #.mean()
                if i == len(train_loader)/2-1:
                    print("MSE: ", train_loss.item())
                train_loss.backward()
                popt.step()
                train_total_loss += train_loss.item()

            train_total_loss /= (len(train_loader))
            print("Average MSE per sample: ", train_total_loss)
            prediction_Net.eval()
            print("\n Entering the validation routine . . .")
            valid_total_loss = 0

            for i, data in enumerate(valid_loader):
                z, c = data['X'], data['Y']
                z, c = z.to(DEVICE), c.to(DEVICE)
                c_hat = prediction_Net(z)
                valid_loss = loss(c_hat, c)

                if i == len(valid_loader) / 2 - 1:
                    print("MSE: ", valid_loss.item())

                valid_total_loss += valid_loss.item()

            valid_total_loss /= (len(valid_loader))
            print("Average MSE per sample: ", valid_total_loss)

            if patience == no_good:
                break
            if valid_total_loss < best:
                best = valid_total_loss
                best_model = copy.deepcopy(prediction_Net)
                no_good = 0
            else:
                scheduler.step()
                no_good += 1

        if 'acopf' in args['probtype']:
            torch.save(best_model.state_dict(),"regressor_model/regressor_"+args['probtype']+"_"+args['acopf_feature_mapping_type']+"_"+str(args["nlayer"])+"_"+str(args['featNet_nlayer'])+".pt")
        else:
            torch.save(best_model.state_dict(), "regressor_model/regressor_" + args['probtype'] + "_" + str(args["nlayer"]) + "_" + str(args['featNet_nlayer'])+"_"+str(args['shift'])+"_"+str(args['scaling']) + ".pt")

        data = {
            'Probtype':[args['probtype']],
            'MSE': [best],
            'nlayer': [args['nlayer']],
            'featNet_nlayer': [args['featNet_nlayer']],
            'shift': [args['shift']],
            'scaling': [args['scaling']],
            'transform_input':[args['transform']],
            'transform_target': [args['transform_target']],
            # 'Status': ['Completed', 'Completed', 'Failed', 'Completed', 'Completed']
        }

        df = panda.DataFrame(data)
        #if args['transform']==1:
        df.to_csv('2S_'+args['probtype']+'_MSE.csv', mode='a', header=False, index=False)
        #else:
	    #df.to_csv('2Stage_regressor_MSE.csv', mode='a', header=False, index=False)

    for i, data in enumerate(test_loader):
        z, c = data['X'], data['Y']
        z, c = z.to(DEVICE), c.to(DEVICE)
        start_time = time.time()
        c_hat = prediction_Net(z)
        end_time = time.time()

    if 'acopf' in args['probtype'] and to_evaluate == True:

        instance = json.load(open("acopf_eval/"+prob_name+"/"+args['acopf_feature_mapping_type']+"/2S_evaluation_gt_"+str(args['nlayer'])+"_"+str(args['featNet_nlayer'])+".json", 'r'))
        instance_c = json.load(open("acopf_eval/"+prob_name+"/"+args['acopf_feature_mapping_type']+"/2S_evaluation_candidate_"+str(args['nlayer'])+"_"+str(args['featNet_nlayer'])+".json", 'r'))

        for i, data in enumerate(test_loader):
            z, c = data['X'], data['Y']
            z, c = z.to(DEVICE), c.to(DEVICE)

        c_hat = prediction_Net(z)

        eq_mean = 0
        eq_max = 0
        ineq_mean = 0
        ineq_max = 0
        infeasible = 0
        n = 833
        for i in range(1, n): #c.shape[0]):
            infy = False

            try:
                pg = torch.tensor(instance_c["pg_candidate"+str(i)])  # with respect to c_hat
            except:
                print("Infeasible problem")
                infeasible += 1
                infy = True

            try:
                pg = torch.tensor(instance["pg_"+str(i)])  # with respect to c
            except:
                print("Infeasible problem")
                infeasible += 1
                infy = True

            if infy == False:
                # tmp = c_hat[i, :]
                if 'demands' in args['acopf_feature_mapping_type'] or 'pascal' in args['acopf_feature_mapping_type']and 'costs' not in args['acopf_feature_mapping_type']:
                    pg = torch.tensor(instance["pg_" + str(i)])  # with respect to c
                    gt_obj = ACOPFPredopt.obj_fn_eval(problem_class, pg)
                    pg = torch.tensor(instance_c["pg_candidate" + str(i)])
                    candidate_obj = ACOPFPredopt.obj_fn_eval(problem_class, pg)
                    avg_regret += abs(candidate_obj - gt_obj) * 100 / abs(gt_obj)
                    qg = torch.tensor(instance["qg_" + str(i)])
                    va = torch.tensor(instance["va_" + str(i)])
                    vm = torch.tensor(instance["vm_" + str(i)])
                    dva = ACOPFProblem.get_dva(problem_class, va)[0, :]
                    pd = c[i, : len_d]  # ['pd']
                    qd = c[i, len_d:]  # ['qd']
                    pd_bus, qd_bus = ACOPFProblem.get_pd_qd_bus(problem_class, pd, qd)
                    ineq_res = ACOPFProblem.ineq_resid_2S(problem_class, vm, dva) #check the last parameter
                    eq_res = ACOPFProblem.eq_resid_2S(problem_class, pd_bus, qd_bus, vm, dva, pg, qg)
                    eq_mean += torch.mean(torch.abs(eq_res))
                    eq_max += torch.max(eq_res) #= max(eq_max, torch.max(eq_res))
                    ineq_mean += torch.mean(ineq_res)
                    ineq_max += torch.max(ineq_res) #max(ineq_max, torch.max(ineq_res))
                elif 'costs' in args['acopf_feature_mapping_type'] and 'demands' not in args['acopf_feature_mapping_type']:
                    pg = torch.tensor(instance["pg_" + str(i)])  # with respect to c
                    gt_obj = ACOPFPredopt.obj_fn_eval_costs(problem_class, pg, c[i,:])
                    pg = torch.tensor(instance_c["pg_candidate" + str(i)])  # with respect to c_hat
                    candidate_obj = ACOPFPredopt.obj_fn_eval_costs(problem_class, pg, c[i,:])
                    avg_regret += abs(candidate_obj - gt_obj) * 100 / abs(gt_obj)

                #else:  #### TO IMPLEMENT


            #print("Eq. mean viol. ",torch.mean(eq_res))
            #print("Eq. max viol. ",torch.max(eq_res))
            #print("Ineq. mean viol. ",torch.mean(ineq_res))
            #print("Ineq. max viol. ",torch.max(ineq_res))

        print("Number of infeasible problem: ", infeasible)
        print("Average regret :", avg_regret/833) #c.shape[0]-infy))
        if 'demands' in args['acopf_feature_mapping_type'] or 'pascal' in args['acopf_feature_mapping_type']:
            print("Eq. mean viol. ", eq_mean/ n) #c.shape[0]-infy))
            print("Eq. max viol. ", eq_max/ n)
            print("Ineq. mean viol. ", ineq_mean/ n) #c.shape[0]-infy))
            print("Ineq. max viol. ", ineq_max/ n)
            eq_mean = eq_mean.detach().cpu().numpy()
            eq_max = eq_max.detach().cpu().numpy()
            ineq_mean = ineq_mean.detach().cpu().numpy()
            ineq_max = ineq_max.detach().cpu().numpy()
        else:
            eq_mean = 0
            eq_max = 0
            ineq_mean = 0
            ineq_max = 0
        data = {
            'Probtype': [args['probtype']],
            'K': [args['featNet_nlayer']],
            'nLayer': [args['nlayer']],
            'Mapping': [args['acopf_feature_mapping_type']],
            'Opt.gap': [avg_regret.detach().cpu().numpy()/ (n-infeasible+1)], #(c.shape[0]-infy)],
            'Max eq. viol': [eq_max / (n-infeasible+1)],
            'Mean eq. viol': [eq_mean /  (n-infeasible+1)], #(c.shape[0]-infy)]
            'Max ineq. viol': [ineq_max /  (n-infeasible+1)],
            'Mean ineq. viol': [ineq_mean /  (n-infeasible+1)],
            'Num. of infeasible instances': [infeasible]
            # 'Status': ['Completed', 'Completed', 'Failed', 'Completed', 'Completed']
        }
        df = panda.DataFrame(data)
        df.to_csv('2s_results.csv', mode='a', header=True, index=False)
    elif args['probtype'] == 'knapsack':
        #x_star_of_c_hat_list = np.array((problem_class.xdim,1000))
        #x_star_of_c_list = np.array((problem_class.xdim,1000))
        for i in range (1000):
            tmp = c_hat[i, :]
            tmp2 = c[i, :]
            x_star_of_c_hat = solve_knapsack(problem_class.A.cpu().numpy(), problem_class.w.cpu().numpy(), tmp.cpu().detach().numpy(),1, problem_class.gamma.cpu().numpy())[0]
            x_star_of_c = solve_knapsack(problem_class.A.cpu().numpy(), problem_class.w.cpu().numpy(), tmp2.cpu().detach().numpy(),1,problem_class.gamma.cpu().numpy())[0]
            gt_obj = KnapsackProblem.obj_fn_eval(problem_class, tmp2, torch.from_numpy(x_star_of_c).to(DEVICE)).cpu().detach().numpy()
            candidate_obj = KnapsackProblem.obj_fn_eval(problem_class, tmp2, torch.from_numpy(x_star_of_c_hat).to(DEVICE)).cpu().detach().numpy()
            avg_regret += abs(candidate_obj-gt_obj)/abs(gt_obj)
            print("Regret per sample: ", abs(candidate_obj-gt_obj)/abs(gt_obj))
            avg_regret_list.append(abs(candidate_obj-gt_obj)/abs(gt_obj))
        avg_regret/=1000
        print("Avg. regret: ",avg_regret)
        print("90% quartile of regret: ", np.quantile(avg_regret_list, [.9]).item())
    elif args['probtype'] == 'portfolio':
        x_star_of_c_hat = solve_convexqp(problem_class.Q.cpu().numpy(), c_hat.T.cpu().detach().numpy(), problem_class.A.cpu().numpy(),
                   problem_class.G.cpu().numpy(), problem_class.h.cpu().numpy(), np.zeros(1))[0]
        x_star_of_c = solve_convexqp(problem_class.Q.cpu().numpy(), c.T.cpu().detach().numpy(), problem_class.A.cpu().numpy(),
                   problem_class.G.cpu().numpy(), problem_class.h.cpu().numpy(), np.zeros(1))[0]
        gt_obj = PortfolioProblem.obj_fn(problem_class, c, torch.from_numpy(x_star_of_c).to(DEVICE)).cpu().detach().numpy()
        candidate_obj = PortfolioProblem.obj_fn(problem_class, c, torch.from_numpy(x_star_of_c_hat).to(DEVICE)).cpu().detach().numpy()
        avg_regret = np.mean(abs(candidate_obj-gt_obj)/abs(gt_obj))
        print("Avg. regret: ",avg_regret)
        tmp = abs(candidate_obj-gt_obj)/abs(gt_obj)
        print("90% quartile of regret", np.quantile(tmp, [.9]).item())
        #avg_regret_list.append((abs(candidate_obj-gt_obj)/abs(gt_obj)).detach().cpu().numpy().item())
    elif args['probtype'] == 'qcqp' :
        avg_regret = 0
        n = 50
        for i in range(n): #len(test_loader)):
            x_star_of_c_hat = solve_qcqp(problem_class.A.cpu().numpy(), c_hat.T[:,i].cpu().detach().numpy())[0]
            x_star_of_c = solve_qcqp(problem_class.A.cpu().numpy(), c.T[:,i].cpu().detach().numpy())[0]
            gt_obj = QCQPProblem.obj_fn_eval(problem_class, c[i,:], torch.from_numpy(x_star_of_c).to(DEVICE)).cpu().detach().numpy()
            candidate_obj = QCQPProblem.obj_fn_eval(problem_class, c[i,:], torch.from_numpy(x_star_of_c_hat).to(DEVICE)).cpu().detach().numpy()
            avg_regret += 1/n*np.mean(abs(candidate_obj-gt_obj)/abs(gt_obj))
            print("Avg. regret: ",avg_regret)
            tmp = abs(candidate_obj-gt_obj)/abs(gt_obj)
            print("90% quartile of regret", np.quantile(tmp, [.9]).item())
        print("Avg. regret: ",avg_regret/len(test_loader))
        data = {
            'Probtype': [args['probtype']],
            'K': [args['featNet_nlayer']],
            'nLayer': [args['nlayer']],
            'Opt.gap': [avg_regret], #(c.shape[0]-infy)],
        }
        df = panda.DataFrame(data)
        df.to_csv('2s_results_w.csv', mode='a', header=True, index=False)

    elif args['probtype'] == 'nonconvexqp':
        n = 200
        start_time_inf = time.time()
        x_star_of_c_hat = solve_nonconvexqp(problem_class.Q.cpu().numpy(), c_hat[:n,:].cpu().detach().numpy(), problem_class.A.cpu().numpy(),
                   problem_class.G.cpu().numpy(), problem_class.h.cpu().numpy(), np.zeros((n,50)))[0]
        end_time_inf = time.time()
        x_star_of_c = solve_nonconvexqp(problem_class.Q.cpu().numpy(), c[:n,:].cpu().detach().numpy(), problem_class.A.cpu().numpy(),
                   problem_class.G.cpu().numpy(), problem_class.h.cpu().numpy(), np.zeros((n,50)))[0]
        gt_obj = NonconvexQPProblem.obj_fn(problem_class, c[:n,:], torch.from_numpy(x_star_of_c).to(DEVICE)).cpu().detach().numpy()
        candidate_obj = NonconvexQPProblem.obj_fn(problem_class, c[:n,:], torch.from_numpy(x_star_of_c_hat).to(DEVICE)).cpu().detach().numpy()
        avg_regret = np.mean(abs(candidate_obj-gt_obj)/abs(gt_obj))
        print("Avg. regret: ",avg_regret)
        tmp = abs(candidate_obj-gt_obj)/abs(gt_obj)
        print("90% quartile of regret", np.quantile(tmp, [.9]).item())
        data = {
            'Probtype': [args['probtype']],
            'Opt.gap': [avg_regret],
            'nLayer': [args['nlayer']],
            'K': [args['featNet_nlayer']],
            'shift': [args['shift']],
            'scaling': [args['scaling']],
            'transform_input': [args['transform']],
            'transform_target': [args['transform_target']],
            'normalize':[args['normalize']]}
        to_be_restored = {}
        for i in range(n):
            to_be_restored[str(i + 1)] = {}
            to_be_restored[str(i + 1)]["2s_regret"] = str((abs(candidate_obj - gt_obj) / abs(gt_obj))[i])
            to_be_restored[str(i + 1)]["sample"] = {}
            to_be_restored[str(i + 1)]["solution"] = {}
            for j in range(c.shape[1]):
                #to_be_restored[str(i + 1)][str(j + 1)] = str(x_star_of_c_hat[i][j])
                to_be_restored[str(i + 1)]["sample"][str(j + 1)] = str(z[i][j].cpu().detach().numpy())
                to_be_restored[str(i + 1)]["solution"][str(j + 1)] = str(x_star_of_c_hat[i][j])
        with open('deep_2S_LTOF_comparison/2s_regret_'+str(args['probtype'])+'_' + str(args['featNet_nlayer']) + '.json', 'w') as fp:
            json.dump(to_be_restored, fp)
	

        #df = panda.DataFrame(data)
        #df.to_csv('2S_'+args['probtype']+'_regret.csv', mode='a', header=False, index=False)
        #df = panda.DataFrame(data_time)
        #df.to_csv('2S_' + args['probtype'] + '_inference_time.csv', mode='a', header=False, index=False)
    elif args['probtype'] == 'bilinear':  ### TO DO
        n = 50
        bilinear_solver = helper_bilinear.BiLinearSolver(10, problem_class.Q, n)
        x_star_of_c_hat = bilinear_solver.solve(c_hat[:n,:].detach())
        x_star_of_c = bilinear_solver.solve(c[:n,:])
        gt_obj = BilinearQPProblem.obj_fn(problem_class, c[:n,:], x_star_of_c[:n,:]).cpu().detach().numpy() # torch.from_numpy(x_star_of_c).to(DEVICE)).cpu().detach().numpy()
        candidate_obj = BilinearQPProblem.obj_fn(problem_class, c[:n,:], x_star_of_c_hat[:n,:]).cpu().detach().numpy() # torch.from_numpy(x_star_of_c_hat).to(DEVICE)).cpu().detach().numpy()
        avg_regret = np.mean(abs(candidate_obj - gt_obj) / abs(gt_obj))
        print("Avg. regret: ", avg_regret)
        tmp = abs(candidate_obj - gt_obj) / abs(gt_obj)
        print("90% quartile of regret", np.quantile(tmp, [.9]).item())
        data = {
            'Probtype': [args['probtype']],
            'Opt.gap': [avg_regret],
            'nLayer': [args['nlayer']],
            'K': [args['featNet_nlayer']],
              # (c.shape[0]-infy)],
            'shift': [args['shift']],
            'scaling': [args['scaling']],
            'transform_input': [args['transform']],
            'transform_target': [args['transform_target']],
            'normalize': [args['normalize']]
        }
        df = panda.DataFrame(data)
        df.to_csv('2S_'+args['probtype']+'_regret.csv', mode='a', header=True, index=False)


def transform(train, valid, test, args):
    scaling_factor = args['scaling']
    shift = args['shift']
    train = (train)/5
    train += torch.ones(train.shape[0], train.shape[1]).to(DEVICE)
    train *= scaling_factor * torch.ones(train.shape[0], train.shape[1]).to(DEVICE)
    valid = (valid/5 + torch.ones(valid.shape[0], valid.shape[1]).to(DEVICE)) * scaling_factor * torch.ones(valid.shape[0], valid.shape[1]).to(DEVICE)
    test = (test/5 + torch.ones(valid.shape[0], test.shape[1]).to(DEVICE)) * scaling_factor * torch.ones(
        test.shape[0], test.shape[1]).to(DEVICE)

    return train, valid, test

def main():
    parser = argparse.ArgumentParser(description='DC3')
    ### PROBLEM TYPE AND METHOD
    parser.add_argument('--probtype', type=str, default='nonconvexqp', choices=['predopt_acopf57', 'portfolio', 'knapsack', 'predopt_acopf118', 'qcqp', 'nonconvexqp', 'bilinear'], help='problem type')
    parser.add_argument('--method', type=str, default='ld', choices=['dc3', 'ld', 'msep', 'pdl'],
                        help='training batch size')
    ### PROBLEM DIMENSION
    parser.add_argument('--nvar', type=int,  help='the number of decision variables')
    parser.add_argument('--nineq', type=int, help='the number of inequality constraints')
    parser.add_argument('--neq', type=int, help='the number of equality constraints')
    parser.add_argument('--nex', type=int, help='total number of data instances')
    ### HYPERPARAMS OF THE PREDICTION NET TRAINING
    parser.add_argument('--epochs', type=int, help='number of neural network epochs')
    parser.add_argument('--seed', type=int, default=1001, help='random seed')
    parser.add_argument('--batchsize', type=int, help='training batch size')
    parser.add_argument('--lr', type=float, help='neural network learning rate')
    ### PDL ARGS
    parser.add_argument('--hiddensize', type=int, default=200,
                        help='hidden layer size for neural network (used for QP and QCQP cases)')
    parser.add_argument('--hiddenfrac', type=float, default=None,
                        help='hidden layer node fraction (only used for ACOPF)')
    parser.add_argument('--use_sigmoid', type=bool, default=False, help='the number of layers')
    parser.add_argument('--rho', type=float, help='initial coefficient of the penalty terms')
    parser.add_argument('--rhomax', type=float, help='maximum rho limit')
    parser.add_argument('--tau', type=float, help='parameter for updating rho')
    parser.add_argument('--alpha', type=float, help='updating rate of rho')
    parser.add_argument('--maxouteriter', type=int, help='maximum outer iterations')
    parser.add_argument('--maxinneriter', type=int, help='maximum inner iterations')
    ### DC3 ARGS
    parser.add_argument('--softweight', type=float, help='total weight given to constraint violations in loss')
    parser.add_argument('--softweighteqfrac', type=float,
                        help='fraction of weight given to equality constraints (vs. inequality constraints) in loss')
    parser.add_argument('--usecompl', type=str_to_bool, help='whether to use completion')
    parser.add_argument('--usetraincorr', type=str_to_bool, help='whether to use correction during training')
    parser.add_argument('--usetestcorr', type=str_to_bool, help='whether to use correction during testing')
    parser.add_argument('--corrmode', choices=['partial', 'full'],
                        help='employ DC3 correction (partial) or naive correction (full)')
    parser.add_argument('--corrtrainsteps', type=int, help='number of correction steps during training')
    parser.add_argument('--corrtestmaxsteps', type=int, help='max number of correction steps during testing')
    parser.add_argument('--correps', type=float, help='correction procedure tolerance')
    parser.add_argument('--corrlr', type=float, help='learning rate for correction procedure')
    parser.add_argument('--corrmomentum', type=float, help='momentum for correction procedure')
    parser.add_argument('--transform', type=int,
                        help='whether to apply a normalization to the neural network input')
    parser.add_argument('--transform_target', type=int,
                        help='whether to apply a normalization to the neural network input')
    ### SUPERVISED ARGS
    parser.add_argument('--losstype', type=str, default='ld', choices=['mae', 'mse', 'maep', 'msep', 'ld'],
                        help='MAE or MSE')
    parser.add_argument('--ldupdatefreq', type=int, help='LD penalty coefficient update epoch frequency')
    parser.add_argument('--ldstepsize', type=float, help='LD multiplier update step size')
    parser.add_argument('--lamg', type=float, help='penalty coefficient for inequality constraints')
    parser.add_argument('--lamh', type=float, help='penalty coefficient for equality constraints')
    parser.add_argument('--normalize', type=bool, default = False, help='whether to apply a normalization to the neural network input')
    parser.add_argument('--nlayer', type=int, default=1, help='the number of layers')
    ### GENERAL ARGS
    parser.add_argument('--nworkers', type=int, default=0, help='the number of workers for dataloader')
    parser.add_argument('--index', type=int, help='index to keep track of different runs')
    parser.add_argument('--objscaler', type=bool, default=None, help='objective scaling factor')
    parser.add_argument('--save', type=bool, default=False, help='whether to save statistics')
    parser.add_argument('--acopf_feature_mapping_type', type=str, default="synthetic_costs")
    parser.add_argument('--featNet_hiddensize', type=int, default=50)
    parser.add_argument('--featsize', type=int, default=30)
    parser.add_argument('--featNet_nlayer', type=int, default=4)
    parser.add_argument('--regressorNet_hidden_size', type=int, default=200)
    parser.add_argument('--regressorNet_nlayer', type=int, default=5)
    parser.add_argument('--regressorNet_init', type=str, default='x',
                        help='initialization of the regressor model')
    parser.add_argument('--regressorNet_dropout_rate', type=float, default=0.1)
    parser.add_argument('--regressorNet_learning_rate', type=float, default=1e-3)
    parser.add_argument('--regressorNet_optimizer', type=int, default=1)
    parser.add_argument('--shift', type=float, default=1.)
    parser.add_argument('--scaling', type=float, default=1.)

    args = parser.parse_args()
    args = vars(args) # change to dictionary
    #print(args['transform'])
    print(CURRENT_PATH)

    set_seed(args['seed'], DEVICE)
    setproctitle('PnO-{}'.format(args['probtype']))

    print("Entering data loading section . . . ")

    if args['probtype'] == 'predopt_acopf57' or args['probtype'] == 'predopt_acopf118':
        if args['method'] == 'pdl':
            args_default = default_args.pdl_default_args(args['probtype'])
            for k, v in args_default.items():
                args[k] = v if args[k] is None else args[k]
            data, args = load_acopf_predopt_data(args, CURRENT_PATH, DEVICE)
        elif args['method'] == 'ld':
            args_default = default_args.baseline_supervised_default_args(args['probtype'], "ld")
            for k, v in args_default.items():
                args[k] = v if args[k] is None else args[k]
            data = load_acopf_predopt_data(args, CURRENT_PATH, DEVICE)
    elif args['probtype'] == 'portfolio':
        if args['method'] == 'pdl':
            args_default = default_args.pdl_default_args(args['probtype'])
            for k, v in args_default.items():
                args[k] = v if args[k] is None else args[k]
            data, args = load_portfolio_data(args, CURRENT_PATH, DEVICE)
        elif args['method'] == 'ld':
            args_default = default_args.baseline_supervised_default_args(args['probtype'], "ld")
            for k, v in args_default.items():
                args[k] = v if args[k] is None else args[k]
            data, args = load_portfolio_data(args, CURRENT_PATH, DEVICE)
        elif args['method'] == 'dc3':
            defaults = default_args.dc3_default_args(args['probtype'])
            for key in defaults.keys():
                if args[key] is None:
                    args[key] = defaults[key]
            data, args = load_portfolio_data(args, CURRENT_PATH, DEVICE)
            #data = load_portfolio_data(args, CURRENT_PATH, DEVICE))
    elif args['probtype'] == 'qcqp':
        if args['method'] == 'pdl':
            args_default = default_args.pdl_default_args(args['probtype'])
            for k, v in args_default.items():
                args[k] = v if args[k] is None else args[k]
            data, args = load_qcqp_data(args, CURRENT_PATH, DEVICE)
        elif args['method'] == 'ld':
            args_default = default_args.baseline_supervised_default_args(args['probtype'], "ld")
            for k, v in args_default.items():
                args[k] = v if args[k] is None else args[k]
            data, args = load_qcqp_data(args, CURRENT_PATH, DEVICE)
    elif args['probtype'] == 'nonconvexqp':
        if args['method'] == 'pdl':
            args_default = default_args.pdl_default_args(args['probtype'])
            for k, v in args_default.items():
                args[k] = v if args[k] is None else args[k]        
            data = load_qp_data(args, CURRENT_PATH, DEVICE)
        elif args['method'] == 'ld':
            args_default = default_args.baseline_supervised_default_args(args['probtype'], "ld")
            for k, v in args_default.items():
                args[k] = v if args[k] is None else args[k]
            data = load_qp_data(args, CURRENT_PATH, DEVICE)
        elif args['method'] == 'dc3':
            defaults = default_args.dc3_default_args(args['probtype'])
            for key in defaults.keys():
                if args[key] is None:
                    args[key] = defaults[key]
            data = load_qp_data(args, CURRENT_PATH, DEVICE)
            #data = load_portfolio_data(args, CURRENT_PATH, DEVICE))
    elif args['probtype'] == 'bilinear':
        if args['method'] == 'pdl':
            args_default = default_args.pdl_default_args(args['probtype'])
            for k, v in args_default.items():
                args[k] = v if args[k] is None else args[k]
            data = load_bilinear_data(args, CURRENT_PATH, DEVICE)
            print("Data loaded.")
        elif args['method'] == 'ld':
            args_default = default_args.baseline_supervised_default_args(args['probtype'], "ld")
            for k, v in args_default.items():
                args[k] = v if args[k] is None else args[k]
            data = load_bilinear_data(args, CURRENT_PATH, DEVICE)
        elif args['method'] == 'dc3':
            defaults = default_args.dc3_default_args(args['probtype'])
            for key in defaults.keys():
                if args[key] is None:
                    args[key] = defaults[key]
            data = load_bilinear_data(args, CURRENT_PATH, DEVICE)
            # data = load_portfolio_data(args, CURRENT_PATH, DEVICE))
    else:
        raise NotImplementedError

    print(args['probtype'])
    print(args['transform'])
    print(args['transform_target'])
    print("Model and data loaded succesfully ")

    print("Defining feature generator model . . .")

    if 'acopf' not in args['probtype']:
        feature_Net = featureNet(args['featNet_nlayer'], args['featNet_hiddensize'], data).to(DEVICE)
        print("Creating prediction net's dataset . . .")
        if args['transform']==1:
            feat_train_target_data, feat_valid_target_data, feat_test_target_data = transform(data.trainX, data.validX, data.testX, args)
        else:
            feat_train_target_data, feat_valid_target_data, feat_test_target_data = data.trainX, data.validX, data.testX
        feature_train_dataset, feature_valid_dataset, feature_test_dataset = feature_Generation(feature_Net, feat_train_target_data, feat_valid_target_data, feat_test_target_data)
    else:
        print("Creating prediction net's dataset . . .")
        if args['acopf_feature_mapping_type'] == 'YT_temp+Y_demand' or 'pascal' in args['acopf_feature_mapping_type']:
            feature_train_dataset, feature_valid_dataset, feature_test_dataset = data.trainX["feat"], data.validX["feat"], data.testX["feat"]
        elif args['acopf_feature_mapping_type'] == 'synthetic_demands':
            #feature_Net = acopf_featureNet(args['featNet_nlayer'], args['featNet_hiddensize'], data.trainX['pd'].shape[1] * 2,args['featsize'])
            feature_train_dataset, feature_valid_dataset, feature_test_dataset = data.trainX["feat"], data.validX["feat"], data.testX["feat"] #feature_Generation(feature_Net, torch.cat((data.trainX['pd'], data.trainX['qd'])), torch.cat((data.validX['pd'], data.validX['qd'])), torch.cat((data.testX['pd'], data.testX['qd'])))
        elif args['acopf_feature_mapping_type'] == 'synthetic_costs':
            #feature_Net = acopf_featureNet(args['featNet_nlayer'], args['featNet_hiddensize'], data.trainX_cost['cost'].shape[1], args['featsize'])
            feature_train_dataset, feature_valid_dataset, feature_test_dataset = data.trainX_cost['feat'], data.validX_cost['feat'], data.testX_cost['feat'] #feature_Generation(feature_Net, data.trainX_cost['cost'], data.validX_cost['cost'], data.testX_cost['cost'])
        elif args['acopf_feature_mapping_type'] == 'synthetic_costs_and_demands_new':
            #feature_Net = acopf_featureNet(args['featNet_nlayer'], args['featNet_hiddensize'],data.trainX_cost['pd'].shape[1] * 2 + data.trainX_cost['cost'].shape[1], args['featsize'])
            feature_train_dataset, feature_valid_dataset, feature_test_dataset = data.trainX_cost['feat'], data.validX_cost['feat'], data.testX_cost['feat'] #feature_Generation(feature_Net,torch.cat((data.trainX_cost['pd'],data.trainX_cost['qd'],data.trainX_cost['cost'] ))
                                                                                                    #,torch.cat(( data.validX_cost['pd'],data.validX_cost['qd'], data.validX_cost['cost'])),
                                                                                                    #torch.cat((data.testX_cost['pd'],data.testX_cost['qd'],data.testX_cost['cost'])))
    if 'acopf' not in args['probtype']:
        if args['transform_target']==1:
            data.trainX, data.validX, data.testX = transform(data.trainX, data.validX, data.testX, args)

        if args['normalize']:
            data.trainX, data.validX, data.testX = normalize(data.trainX.cpu().detach()), normalize(data.validX.cpu().detach()), normalize(data.testX.cpu().detach())

        train_labels = np.array(data.trainX) #.cpu().detach().numpy())#, dtype=np.float32)
        valid_labels = np.array(data.validX) #.cpu().detach().numpy())
        test_labels = np.array(data.testX) #.cpu().detach().numpy())

    elif args['acopf_feature_mapping_type'] == 'synthetic_demands' or args['acopf_feature_mapping_type'] == 'YT_temp+Y_demand' or 'pascal' in args['acopf_feature_mapping_type']:
        a = data.trainX["pd"].cpu().detach().numpy()
        b = data.trainX["qd"].cpu().detach().numpy()
        t = np.concatenate((a,b), axis=1)
        train_labels = np.array(t)

        a = data.validX["pd"].cpu().detach().numpy()
        b = data.validX["qd"].cpu().detach().numpy()
        t = np.concatenate((a, b), axis=1)
        valid_labels = np.array(t)

        a = data.testX["pd"].cpu().detach().numpy()
        b = data.testX["qd"].cpu().detach().numpy()
        t = np.concatenate((a, b), axis=1)
        test_labels = np.array(t)

    elif args['acopf_feature_mapping_type'] == 'synthetic_costs':
        a = data.trainX_cost["cost"][:,1,:].cpu().detach().numpy()
        a = normalize(a) #[normalize(i) for i in a]
        train_labels = np.array(a)
        a = data.validX_cost["cost"][:,1,:].cpu().detach().numpy()
        valid_labels = np.array(a)
        a = data.testX_cost["cost"][:,1,:].cpu().detach().numpy()
        a = normalize(a)
        test_labels = np.array(a)
    else:
        a = data.trainX_cost["pd"].cpu().detach().numpy()
        b = data.trainX_cost["qd"].cpu().detach().numpy()
        c = data.trainX_cost["cost"][:,1,:].cpu().detach().numpy()
        c = normalize(c)

        t = np.concatenate((a, b, c), axis=1)
        train_labels = np.array(t)

        a = data.validX_cost["pd"].cpu().detach().numpy()
        b = data.validX_cost["qd"].cpu().detach().numpy()
        c = data.validX_cost["cost"][:,1,:].cpu().detach().numpy()
        c = normalize(c)
        t = np.concatenate((a, b, c), axis=1)
        valid_labels = np.array(t)

        a = data.testX_cost["pd"].cpu().detach().numpy()
        b = data.testX_cost["qd"].cpu().detach().numpy()
        c = data.testX_cost["cost"][:,1,:].cpu().detach().numpy()
        c = normalize(c)
        t = np.concatenate((a, b, c), axis=1)
        test_labels = np.array(t)

    train_data = np.array(feature_train_dataset.cpu().detach().numpy())  # , dtype=np.float32)
    prediction_Net_train_dataset = D(train_data, train_labels)
    valid_data = np.array(feature_valid_dataset.cpu().detach().numpy()) #, dtype=np.float32)
    prediction_Net_valid_dataset = D(valid_data, valid_labels)
    test_data = np.array(feature_test_dataset.cpu().detach().numpy()) #, dtype=np.float32)
    prediction_Net_test_dataset = D(test_data, test_labels)

    train_loader = DataLoader(prediction_Net_train_dataset, batch_size=args['batchsize'], shuffle=False, num_workers=args['nworkers'])
    valid_loader = DataLoader(prediction_Net_valid_dataset, batch_size=args['batchsize'], shuffle=False, num_workers=args['nworkers'])
    test_loader = DataLoader(prediction_Net_test_dataset, batch_size=len(prediction_Net_test_dataset), shuffle=False, num_workers=args['nworkers'])

    print("Dataset created.")

    feature_dim = feature_train_dataset.shape[1]

    if 'acopf' not in args['probtype']:
        param_dim = data.trainX.shape[1]
    elif args['acopf_feature_mapping_type'] == 'synthetic_demands' or args['acopf_feature_mapping_type'] == 'YT_temp+demand'  or 'pascal' in args['acopf_feature_mapping_type'] :
        param_dim = 2 * data.trainX['pd'].shape[1]
    elif args['acopf_feature_mapping_type'] == 'synthetic_costs':
        param_dim = data.trainX_cost['cost'].shape[2]
    else:
        param_dim = data.trainX_cost['cost'].shape[2] + 2 * data.trainX['pd'].shape[1]

    print("Defining prediction model . . .")
    prediction_Net = predNet(feature_dim, param_dim, args).to(DEVICE)
    # #TVT(prediction_Net, train_loader, valid_loader, test_loader, proxy, args, data)
    TVT(prediction_Net, train_loader, valid_loader, test_loader, args, data)
    print("Training is done.")


if __name__ == '__main__':
    main()



