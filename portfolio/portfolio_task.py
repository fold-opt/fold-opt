import torch
import torch.nn as nn
import numpy as np
import pickle
import math
from portfolio_task_solver import get_markowitz_constraints_cvx, solve_markowitz_cvx
from portfolio_task_utils import spo_grad_cvx, test_regret, train_fwdbwd_spo_cvx, BlackboxMarkowitzWrapper, train_fwdbwd_blackbox_cvx, train_fwdbwd_SQP
import argparse
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from unrolled_ops import DiffSQP
from portfolio_train_utils import PortfolioDiffSQP

from portfolio_train_utils import PortfolioDiffSQP, gurobiBatchSolver, get_sqp_layer
from fold_opt.fold_opt import FoldOptLayer


parser = argparse.ArgumentParser()

parser.add_argument("--n_samples", type=int, default=320)
parser.add_argument("--n", type=int, default=50)
parser.add_argument("--p", type=int, default=5)
parser.add_argument("--tau", type=float, default=0.1)
parser.add_argument("--deg", type=int, default=1)

parser.add_argument('--batsize', type=int, default=32)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--blackbox_lambda', type=float, default=15.0)
parser.add_argument('--load_dataset', type=str, default='noname.txt')
parser.add_argument('--train_mode', type=str, default='spo')
parser.add_argument('--output_tag', type=str, default='notag')
parser.add_argument('--index', type=int, default=9999)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)


# Generate data instances
n_samples = args.n_samples
n = args.n    # number of assets
p = args.p    # number of features
tau = args.tau  # noise level parameter
deg = args.deg  # degree parameter
# Bernoulli(0.5) matrix
# parameters of 'true' model
B = np.int_(np.random.rand(n,p) > 0.5)
# load factor matrix
# 50 and 4 are specified in the paper
L =   2*0.0025*tau*np.random.rand(n,4) - 0.0025*tau
pairs = []
for _ in range(0,n_samples):
    x = np.random.normal(0, 1, size=p)    # feature vector - standard normal
    r = (  (0.05/math.sqrt(p)) * np.matmul(B,x) + (0.1)**(1/deg)  )**(deg)
    f = np.random.normal(0, 1, size=4)
    eps = np.random.normal(0, 1, size=n)
    r = r + np.matmul(L,f) + 0.01*tau*eps
    c = r
    pairs.append(  (x,c)  )
# This COV should be made from the same L in data generation and model def
COV = np.matmul(L,L.T)+ np.eye(n)*(0.01*tau)**2   # covariance matrix

w_ = np.ones(n)/n     # JK check is this correct?   p.29 'e denotes the vector of all ones' - does this answer it?
gamma = 22.5/2 * np.matmul( np.matmul(w_,COV), w_ )

COV   = 1e6*COV
gamma = 1e6*gamma

print("gamma")
print( gamma )
print("COV")
print( COV )
print("np.matmul( np.matmul(w_,COV), w_ )")
print( np.matmul( np.matmul(w_,COV), w_ ) )
print("(w_@COV)@w_")
print( (w_@COV)@w_ )
input()

# Arrange datasets
inputs  = torch.stack(  [ torch.Tensor(a) for (a,b) in pairs ]  )
targets = torch.stack(  [ torch.Tensor(b) for (a,b) in pairs ]  )

cutoff = int( n_samples*0.80 )

train_inputs  = inputs[:cutoff]
train_targets = targets[:cutoff]

test_inputs   = inputs[cutoff:]
test_targets  = targets[cutoff:]


# model input size: p (number of features) -> default 5
# model output size: n (number of assets)  -> default 50
model = torch.nn.Sequential( nn.Linear(p,n), torch.nn.ReLU(), torch.nn.BatchNorm1d(n), nn.Linear(n,2*n), torch.nn.BatchNorm1d(2*n),  torch.nn.ReLU(), nn.Linear(2*n,n), torch.nn.BatchNorm1d(n),  torch.nn.ReLU(), nn.Linear(n,n) )
optimizer = torch.optim.Adam( model.parameters(), lr=args.lr  )

train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batsize)

test_dataset = torch.utils.data.TensorDataset(test_inputs, test_targets)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batsize) # might not need this

constraints, variables = get_markowitz_constraints_cvx(n,p,tau,L)

test_regret(model, test_inputs, test_targets, constraints, variables)


if args.train_mode == 'blackbox':
    blackbox_layer = BlackboxMarkowitzWrapper(constraints, variables, args.blackbox_lambda)()


if args.train_mode == 'SQP':
    COV_t = torch.Tensor(COV).double()
    quadreg = 0.1
    fwd_solver = gurobiBatchSolver(n,COV,gamma,quadreg, return_dual=True).solve
    sqp_layer  = get_sqp_layer(n,COV,gamma,quadreg, return_dual = True, alpha=0.01, max_iter=1)
    update_step = lambda p,x: sqp_layer(p, primal0=x[:,:n], dual0=x[:,n:])
    SQPFolded = FoldOptLayer(fwd_solver, update_step, n_iter=n+1, backprop_rule='GMRES')
    SQP_layer = lambda c: SQPFolded(c)[:,:n].float()



print("train_mode:")
print(args.train_mode)

train_regrets = []
test_regrets  = []
batch_regrets = []
for epoch in range(0,args.epochs):
    print("Entering Epoch {}".format(epoch))
    epoch_regrets = [] # finish getting regret by epoch
    for batch_idx, (input, target) in enumerate(train_loader):

        print("Batch {}".format(batch_idx))

        if args.train_mode == 'spo':
            regret = train_fwdbwd_spo_cvx(model, optimizer, constraints, variables, input, target)

        elif args.train_mode == 'blackbox':
            regret = train_fwdbwd_blackbox_cvx(model, optimizer, blackbox_layer, input, target)

        elif args.train_mode == 'unrolled_SQP':
            regret = train_fwdbwd_SQP(model, optimizer, SQP_layer, input, target, COV_t, gamma)

        elif args.train_mode == 'SQP':
            regret = train_fwdbwd_SQP(model, optimizer, SQP_layer, input, target, COV_t, gamma)


        mean_regret = regret.mean().item()
        epoch_regrets.append( mean_regret )
        print("regret = {}".format(mean_regret))

        batch_regrets.append(mean_regret)

    print("test regret  = {}".format(test_regret(model, test_inputs, test_targets, constraints, variables).mean().item()))
    print("train regret = {}".format( torch.mean(torch.Tensor(epoch_regrets)).item() ) )


    test_regrets.append( test_regret(model, test_inputs, test_targets, constraints, variables).mean().item() )
    train_regrets.append( torch.mean(torch.Tensor(epoch_regrets)).item() )


plt.semilogy(range(len(test_regrets)), test_regrets, label='Avg Regret Test')
plt.semilogy(range(len(train_regrets)), train_regrets, label='Avg Regret Train')
plt.xlabel('Training Epoch')
plt.ylabel('Avg Regret')
plt.ylim(-0.05, 0.2)
plt.show()

portfolio_dump = {}
portfolio_dump['index']  = args.index
portfolio_dump['n']      = args.n
portfolio_dump['deg']    = args.deg
portfolio_dump['mode']   = args.train_mode
portfolio_dump['epoch_regrets'] = epoch_regrets
portfolio_dump['batch_regrets'] = batch_regrets
pickle.dump( portfolio_dump,     open('portfolio_dump_'+ str(args.index)+ '.p','wb') )


csv_outs = {}

csv_outs["n_samples"] = args.n_samples
csv_outs["n"] = args.n
csv_outs["p"] = args.p
csv_outs["tau"] = args.tau
csv_outs["deg"] = args.deg

csv_outs["batsize"] = args.batsize
csv_outs["epochs"] = args.epochs
csv_outs["lr"] = args.lr
csv_outs["seed"] = args.seed
csv_outs["blackbox_lambda"] = args.blackbox_lambda
csv_outs["load_dataset"] = args.load_dataset
csv_outs["train_mode"] = args.train_mode
csv_outs["output_tag"] = args.output_tag
csv_outs["index"] = args.index
csv_outs["n_samples"] = n_samples
csv_outs["n"] = n
csv_outs["p"] = p
csv_outs["tau"] = tau
csv_outs["deg"] = deg
csv_outs["gamma"] = gamma
csv_outs["train_regrets_final"] = train_regrets[-1]
csv_outs["test_regrets_final"]  = test_regrets[-1]



csv_outs = {k:[v] for (k,v) in csv_outs.items()   }
df_outs = pd.DataFrame.from_dict(csv_outs)
outPathCsv = './csv/'+ "portfl" + args.output_tag + "_run_" + str(args.index) + ".csv"
df_outs.to_csv(outPathCsv)
