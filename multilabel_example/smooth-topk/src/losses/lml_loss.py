import torch
from torch import nn


from multilabel_example.multilabel_models import EntropyKnapsackPGD, EntropyKnapsackSQP
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import cvxpy as cp

from multilabel_example.lml import LML

class LMLLoss(nn.Module):
    def __init__(self, n_classes, k=5, tau=1.0):
        super(LMLLoss, self).__init__()
        self.n_classes = n_classes
        self.k = k
        self.tau = tau

    def forward(self, x, y):
        n_batch = x.shape[0]
        x = nn.functional.normalize(x)

        p = LML(N=self.k, eps=1e-4)(x/self.tau)
        losses = -torch.log(p.gather(1, y.unsqueeze(1)) + 1e-8)
        return losses.mean()



class PGDLoss(nn.Module):
    def __init__(self, n_classes, k=5, tau=1.0, alpha = 0.025, n_iter = 80):
        super(PGDLoss, self).__init__()
        self.n_classes = n_classes
        self.k = k
        self.tau = tau
        self.entknap_fixed = EntropyKnapsackPGD(n_classes, k, stepsize = alpha, n_iter=n_iter).solve

    def forward(self, x, y):
        n_batch = x.shape[0]
        x = nn.functional.normalize(x)
        p = self.entknap_fixed(x/self.tau)
        losses = -torch.log(p.gather(1, y.unsqueeze(1)) + 1e-8)
        return losses.mean()
    


class SQPLoss(nn.Module):
    def __init__(self, n_classes, k=5, tau=1.0):
        super(SQPLoss, self).__init__()
        self.n_classes = n_classes
        self.k = k
        self.tau = tau
        self.entknap_fixed = EntropyKnapsackSQP(n_classes, k).solve

    def forward(self, x, y):
        n_batch = x.shape[0]
        x = nn.functional.normalize(x)
        p = self.entknap_fixed(x/self.tau)
        losses = -torch.log(p.gather(1, y.unsqueeze(1)) + 1e-8)
        return losses.mean()



class CVXLoss(nn.Module):
    def __init__(self, n_classes, k=5, tau=1.0, eps=1.0):
        super(CVXLoss, self).__init__()
        self.n_classes = n_classes
        self.k = k
        self.tau = tau

        z = cp.Variable(n_classes)
        e = cp.Parameter(n_classes)
        #constraints = [G@x<=h, A@x==b]
        constraints = [0<=z, z<=1, cp.sum(z)==k]
        objective = cp.Maximize( e@z + eps*(cp.sum(cp.entr(z)) ) )
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        entro_knapsack_cvxlayer = CvxpyLayer(problem, parameters=[e], variables=[z])
        self.entro_knapsack_cvx = lambda z: entro_knapsack_cvxlayer(z)[0]

    def forward(self, x, y):
        n_batch = x.shape[0]
        x = nn.functional.normalize(x)
        p = self.entro_knapsack_cvx(x/self.tau)
        losses = -torch.log(p.gather(1, y.unsqueeze(1)) + 1e-8)
        return losses.mean()
    