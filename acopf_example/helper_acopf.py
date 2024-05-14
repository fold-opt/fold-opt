import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from functools import reduce
import json, operator
# JK
import random
import os
from torch.utils.data import Dataset
import numpy as np
import casadi as ca


class ACOPFProblem:
    def __init__(self, filepath, acopf_name, device, valid_frac=0.0833, test_frac=0.0833, obj_scaler=1e5):
        filepath = Path(filepath)
        self.acopf_name = acopf_name
        self.device = device
        self.valid_frac = valid_frac
        self.test_frac = test_frac
        self.train_frac = 1.-valid_frac-test_frac

        # laod network file
        network = json.load(open(filepath/"network.json"))
        self.nbus = len(network["bus"])
        self.ngen = len(network["gen"])
        self.nload = len(network["load"])
        self.nbranch = len(network["branch"])
        self.nshunt = len(network["shunt"])
        self.loadids = np.sort(np.array(list(network["load"].keys()),dtype=np.int64))
        self.genids = np.sort(np.array(list(network["gen"].keys()),dtype=np.int64))
        self.busids = np.sort(np.array(list(network["bus"].keys()),dtype=np.int64))
        self.branchids = np.sort(np.array(list(network["branch"].keys()),dtype=np.int64))
        self.shuntids = np.sort(np.array(list(network["shunt"].keys()),dtype=np.int64)) # note that id starts from 1 typically

        # look up slack bus
        self.slack_bus_idx = []
        for bus_id, bus_data in network["bus"].items():
            if bus_data['bus_type'] == 3:
                bus_idx = np.where(self.busids==int(bus_id))[0][0]
                self.slack_bus_idx.append(bus_idx)
        assert len(self.slack_bus_idx) == 1
        self.slack_bus_idx = self.slack_bus_idx[0]

        self.baseMVA = network["baseMVA"]
        self.obj_scaler = obj_scaler

        self.quad_cost, self.lin_cost, self.const_cost = np.zeros(self.ngen),np.zeros(self.ngen),np.zeros(self.ngen)
        self.pgmin, self.pgmax, self.qgmin, self.qgmax = [], [], [], []
        self.gen2bus = [] # gen index to bus index

        # gen
        for i,id in enumerate(self.genids):
            gen_data = network["gen"][str(id)]
            cost_list = gen_data["cost"]
            for idx, c in enumerate(cost_list[::-1]):
                if idx == 0:
                    self.const_cost[i] = c
                elif idx == 1:
                    self.lin_cost[i] = c
                elif idx == 2:
                    self.quad_cost[i] = c
                else:
                    assert False
            self.pgmin.append(gen_data["pmin"]); self.qgmin.append(gen_data["qmin"]); self.pgmax.append(gen_data["pmax"]); self.qgmax.append(gen_data["qmax"])
            bus_id = gen_data["gen_bus"]
            bus_idx = np.where(self.busids==int(bus_id))[0][0]
            self.gen2bus.append(bus_idx)

        self.pgmin = np.array(self.pgmin)#.to(self.device)
        self.pgmax = np.array(self.pgmax)#.to(self.device)
        self.qgmin = np.array(self.qgmin)#.to(self.device)
        self.qgmax = np.array(self.qgmax)#.to(self.device)

        # bus
        self.vmmax, self.vmmin = [], []
        self.basekv = []
        for i,id in enumerate(self.busids):
            bus_data = network["bus"][str(id)]
            self.vmmax.append(bus_data["vmax"]); self.vmmin.append(bus_data["vmin"])
            self.basekv.append(bus_data["base_kv"])

        self.vmmax = np.array(self.vmmax)#.to(self.device)
        self.vmmin = np.array(self.vmmin)#.to(self.device)
        self.basekv = np.array(self.basekv)#.to(self.device)

        # setup bus_genidxs
        bus_genidxs = []
        max_ngens = 0 # max num of generators at bus
        for i,id in enumerate(self.busids):
            genidxs = []
            for gen_idx, bus_idx in enumerate(self.gen2bus):
                if i==bus_idx:
                    genidxs.append(gen_idx)
            bus_genidxs.append(genidxs)
            max_ngens = max(max_ngens, len(genidxs))

        self.bus_genidxs = self.ngen*np.ones((self.nbus,max_ngens), dtype=np.int64)#.to(device)
        for i, genidxs in enumerate(bus_genidxs):
            for j,genidx in enumerate(genidxs):
                self.bus_genidxs[i,j] = genidx

        # load
        self.load2bus = [] # load idx to bus idx
        for i,id in enumerate(self.loadids):
            load_data = network["load"][str(id)]
            bus_id = load_data["load_bus"]
            # print(self.busids)
            # print(bus_id)
            bus_idx = np.where(self.busids==int(bus_id))[0][0]
            self.load2bus.append(bus_idx)

        # branch
        br_r, br_x = [], []
        tap, shift = [], [] # related to the transformer
        self.g_to, self.g_fr = [], []
        self.b_to, self.b_fr = [], []
        self.angmin, self.angmax = [], []
        self.bus_i, self.bus_j = [], [] # bus_i == f_bus | bus_j == t_bus || branch_out_per_bus==bus_i | branch_in_per_bus==bus_j #busidx (not id)
        self.thermal_limit = []
        self.edges = [] # i, j, and branch_id

        for i,id in enumerate(self.branchids):
            branch_data = network["branch"][str(id)]
            br_r.append(branch_data["br_r"]); br_x.append(branch_data['br_x'])
            tap.append(branch_data["tap"]); shift.append(branch_data["shift"])
            self.g_to.append(branch_data["g_to"]); self.g_fr.append(branch_data["g_fr"])
            self.b_to.append(branch_data["b_to"]); self.b_fr.append(branch_data["b_fr"])
            self.angmin.append(branch_data["angmin"]); self.angmax.append(branch_data["angmax"])
            self.thermal_limit.append(branch_data["rate_a"]) # only rate_a is considered in AC-OPF
            bus_i_id = branch_data["f_bus"]; bus_j_id = branch_data["t_bus"]
            bus_i_idx = np.where(self.busids==int(bus_i_id))[0][0]; bus_j_idx = np.where(self.busids==int(bus_j_id))[0][0]
            self.bus_i.append(bus_i_idx); self.bus_j.append(bus_j_idx)
            self.edges.append((self.bus_i[-1], self.bus_j[-1], {"idx": i}))

        br_r = np.array(br_r)#.to(self.device)
        br_x = np.array(br_x)#.to(self.device)
        tap = np.array(tap)#.to(self.device)
        shift = np.array(shift)#.to(self.device)
        self.g_to = np.array(self.g_to)#.to(self.device)
        self.g_fr = np.array(self.g_fr)#.to(self.device)
        self.b_to = np.array(self.b_to)#.to(self.device)
        self.b_fr = np.array(self.b_fr)#.to(self.device)
        self.angmin = np.array(self.angmin)#.to(self.device) # radian
        self.angmax = np.array(self.angmax)#.to(self.device) # radian
        self.bus_i = np.array(self.bus_i)#.to(self.device)
        self.bus_j = np.array(self.bus_j)#.to(self.device)
        self.thermal_limit = np.array(self.thermal_limit)#.to(self.device)

        assert self.thermal_limit.max()>0.
        # assert br_r.size() == br_x.size() and br_x.size() == self.g_to.size()
        # assert self.g_to.size() == self.g_fr.size() and self.g_fr.size()[0] == self.nbranch

        br_r2_x2 = np.power(br_r, 2) + np.power(br_x, 2)
        self.br_g = br_r/br_r2_x2
        self.br_b = -br_x/br_r2_x2
        self.tap2 = np.power(tap, 2)
        self.T_R = tap * np.cos(shift)
        self.T_I = tap * np.sin(shift)

        # setup bus_branchidxs_fr, bus_branchidxs_to
        bus_branchidxs_fr, bus_branchidxs_to = [], []
        max_fr_nbranches, max_to_nbranches = 0, 0
        for i,id in enumerate(self.busids):
            branchidxs_fr, branchidxs_to = [], []
            for branch_idx, branch_bus_idx in enumerate(self.bus_i): # for from busidxs
                if i==branch_bus_idx:
                    branchidxs_fr.append(branch_idx)
            bus_branchidxs_fr.append(branchidxs_fr)
            max_fr_nbranches = max(max_fr_nbranches, len(branchidxs_fr))

            for branch_idx, branch_bus_idx in enumerate(self.bus_j): # for to busidxs
                if i==branch_bus_idx:
                    branchidxs_to.append(branch_idx)
            bus_branchidxs_to.append(branchidxs_to)
            max_to_nbranches = max(max_to_nbranches, len(branchidxs_to))

        self.bus_branchidxs_fr = self.nbranch*np.ones((self.nbus,max_fr_nbranches),dtype=np.int64)#.to(device)
        self.bus_branchidxs_to = self.nbranch*np.ones((self.nbus,max_to_nbranches),dtype=np.int64)#.to(device)
        for i,branchidxs_fr in enumerate(bus_branchidxs_fr):
            for j,branchidx in enumerate(branchidxs_fr):
                self.bus_branchidxs_fr[i,j] = branchidx
        for i,branchidxs_to in enumerate(bus_branchidxs_to):
            for j,branchidx in enumerate(branchidxs_to):
                self.bus_branchidxs_to[i,j] = branchidx

        # shunt
        self.gs = np.zeros(self.nbus)#.to(device)
        self.bs = np.zeros(self.nbus)#.to(device)

        for i,id in enumerate(self.shuntids):
            shunt_data = network["shunt"][str(id)]
            shunt_bus_id = shunt_data["shunt_bus"]
            shunt_bus_idx = np.where(self.busids==int(shunt_bus_id))[0][0]
            gs = shunt_data["gs"]
            bs = shunt_data["bs"]
            self.gs[shunt_bus_idx] += gs
            self.bs[shunt_bus_idx] += bs

        self.gs = np.array(self.gs)
        self.bs = np.array(self.bs)

        # data loading -- to cpu
        print(" Loading Data Instances...",flush=True)
        datapath = filepath/"data"
        datafiles = list(datapath.glob("*.json"))
        self.ndata = len(datafiles)
        self.va = np.empty((self.ndata,self.nbus)) # ground truth - voltage angle
        self.vm = np.empty((self.ndata,self.nbus)) # ground truth - voltage magnitude
        self.pg = np.empty((self.ndata,self.ngen)) # ground truth - active power generation
        self.qg = np.empty((self.ndata,self.ngen)) # ground truth - reactive power generation
        self.pd = np.empty((self.ndata,self.nload)) # input - active demand
        self.qd = np.empty((self.ndata,self.nload)) # input - reactive demand
        self.objs = np.empty(self.ndata)

        instance0 = json.load(open(datafiles[0],'r'))
        self.nfeat = len(instance0["feat"])
        self.feat = np.empty((self.ndata,self.nfeat))

        for i, datafile in enumerate(datafiles):
            instance = json.load(open(datafile,'r'))
            self.va[i,:] = np.array(instance["va"])   # voltage
            self.vm[i,:] = np.array(instance["vm"])   # voltage
            self.pg[i,:] = np.array(instance["pg"])   # active generation
            self.qg[i,:] = np.array(instance["qg"])   # inactive generation
            self.pd[i,:] = np.array(instance["pd"])   # active demand
            self.qd[i,:] = np.array(instance["qd"])   # inactive demand
            self.objs[i] = instance["obj"] # just for reference
            self.feat[i,:] = torch.tensor(instance["feat"]) # observed features (which map to demand)

        self.va = self.va
        self.vm = self.vm
        self.pg = self.pg
        self.qg = self.qg
        self.pd = self.pd
        self.qd = self.qd
        self.dva = self.va[:,self.bus_i] - self.va[:,self.bus_j]

        pd_extended = np.zeros((self.ndata,self.nload,self.nbus))
        qd_extended = np.zeros((self.ndata,self.nload,self.nbus))
        pd_extended[:,np.arange(self.nload),self.load2bus] = self.pd
        qd_extended[:,np.arange(self.nload),self.load2bus] = self.qd
        self.pd_bus = pd_extended.sum(axis=1)
        self.qd_bus = qd_extended.sum(axis=1)

        self.neq = 2*self.nbus # power balance (p and q)
        self.nineq = 2*self.nbranch

        self.pad = lambda x: np.pad(x, ((0,0) ,(0,1)),constant_values=0.,mode='constant') # for padding flow to recover bus idx based tensor # used in 'compute_flow'

    def __str__(self):
        return self.acopf_name

    @property
    def X(self):
        #return {"feat":self.feat}
        return {"feat":self.feat,"pd":self.pd, "qd":self.qd, "pd_bus":self.pd_bus, "qd_bus":self.qd_bus}

    @property
    def trainX(self):
        #return {"feat":self.feat[:int(self.ndata * self.train_frac)]}
        return {"feat":self.feat[:int(self.ndata * self.train_frac)], "pd":self.pd[:int(self.ndata * self.train_frac)], "qd":self.qd[:int(self.ndata * self.train_frac)],
                "pd_bus":self.pd_bus[:int(self.ndata * self.train_frac)], "qd_bus":self.qd_bus[:int(self.ndata * self.train_frac)]}

    @property
    def validX(self):
        #return {"feat":self.feat[int(self.ndata * self.train_frac):int(self.ndata * (self.train_frac + self.valid_frac))]}
        return {"feat":self.feat[int(self.ndata * self.train_frac):int(self.ndata * (self.train_frac + self.valid_frac))],
                "pd":self.pd[int(self.ndata * self.train_frac):int(self.ndata * (self.train_frac + self.valid_frac))],
                "qd":self.qd[int(self.ndata * self.train_frac):int(self.ndata * (self.train_frac + self.valid_frac))],
                "pd_bus":self.pd_bus[int(self.ndata * self.train_frac):int(self.ndata * (self.train_frac + self.valid_frac))],
                "qd_bus":self.qd_bus[int(self.ndata * self.train_frac):int(self.ndata * (self.train_frac + self.valid_frac))]}

    @property
    def testX(self):
        #return {"feat":self.feat[int(self.ndata * (self.train_frac + self.valid_frac)):]}
        return {"feat":self.feat[int(self.ndata * (self.train_frac + self.valid_frac)):],
                "pd":self.pd[int(self.ndata * (self.train_frac + self.valid_frac)):],
                "qd":self.qd[int(self.ndata * (self.train_frac + self.valid_frac)):],
                "pd_bus":self.pd_bus[int(self.ndata * (self.train_frac + self.valid_frac)):],
                "qd_bus":self.qd_bus[int(self.ndata * (self.train_frac + self.valid_frac)):]}

    @property
    def Y(self):
        return {"va":self.va, "vm":self.vm, "pg":self.pg, "qg":self.qg, "dva": self.dva}

    @property
    def trainY(self):
        return {"va":self.va[:int(self.ndata*self.train_frac)],
                "vm":self.vm[:int(self.ndata*self.train_frac)],
                "pg":self.pg[:int(self.ndata*self.train_frac)],
                "qg":self.qg[:int(self.ndata*self.train_frac)],
                "dva":self.dva[:int(self.ndata*self.train_frac)]
                }

    @property
    def validY(self):
        return {"va":self.va[int(self.ndata*self.train_frac):int(self.ndata*(self.train_frac + self.valid_frac))],
                "vm":self.vm[int(self.ndata*self.train_frac):int(self.ndata*(self.train_frac + self.valid_frac))],
                "pg":self.pg[int(self.ndata*self.train_frac):int(self.ndata*(self.train_frac + self.valid_frac))],
                "qg":self.qg[int(self.ndata*self.train_frac):int(self.ndata*(self.train_frac + self.valid_frac))],
                "dva":self.dva[int(self.ndata*self.train_frac):int(self.ndata*(self.train_frac + self.valid_frac))]
                }

    @property
    def testY(self):
        return {"va":self.va[int(self.ndata*(self.train_frac + self.valid_frac)):],
                "vm":self.vm[int(self.ndata*(self.train_frac + self.valid_frac)):],
                "pg":self.pg[int(self.ndata*(self.train_frac + self.valid_frac)):],
                "qg":self.qg[int(self.ndata*(self.train_frac + self.valid_frac)):],
                "dva":self.dva[int(self.ndata*(self.train_frac + self.valid_frac)):]
                }

    def compute_flow(self, vm, dva, verbose=False):
        vmi = vm[:,self.bus_i]
        vmj = vm[:,self.bus_j]
        vmi2 = np.power(vmi, 2)
        vmj2 = np.power(vmj, 2)
        vmij = vmi*vmj

        vaij_cos = np.cos(dva)
        vaij_sin = np.sin(dva)

        pf_fr = (1/self.tap2) * (self.br_g + self.g_fr) * vmi2\
                            + ((-self.br_g * self.T_R + self.br_b * self.T_I)/self.tap2) * (vmij) * vaij_cos\
                            + ((-self.br_b * self.T_R - self.br_g * self.T_I)/self.tap2) * (vmij) * vaij_sin
        pf_to = (self.br_g + self.g_to) * vmj2\
                            + ((-self.br_g * self.T_R - self.br_b * self.T_I)/self.tap2) * (vmij) * vaij_cos\
                            + ((-self.br_b * self.T_R + self.br_g * self.T_I)/self.tap2) * (vmij) * (-vaij_sin)
        qf_fr = - (1/self.tap2) * (self.br_b + self.b_fr) * vmi2\
                            - ((-self.br_b * self.T_R - self.br_g * self.T_I)/self.tap2) * (vmij) * vaij_cos\
                            + ((-self.br_g * self.T_R + self.br_b * self.T_I)/self.tap2) * (vmij) * vaij_sin
        qf_to = -(self.br_b + self.b_to) * vmj2\
                            - ((-self.br_b * self.T_R + self.br_g * self.T_I)/self.tap2) * (vmij) * vaij_cos\
                            + ((-self.br_g * self.T_R - self.br_b * self.T_I)/self.tap2) * (vmij) * (-vaij_sin)
        # print('x0: ', pf_fr.shape)

        if verbose:
            print("compute_flow check params:: %.4f | %.4f | %.4f | %.4f"%(self.tap2.max(), self.br_g.max(), self.g_fr.max(), vmi2.max()),flush=True)
            term1 = (1/self.tap2) * (self.br_g + self.g_fr) * vmi2
            term2 = ((-self.br_g * self.T_R + self.br_b * self.T_I)/self.tap2) * (vmij) * vaij_cos
            term3 = ((-self.br_b * self.T_R - self.br_g * self.T_I)/self.tap2) * (vmij) * vaij_sin
            print("compute_flow:: %.4f | %.4f | %.4f "%(term1.max(), term2.max(), term3.max()),flush=True)

        pf_fr_pad = self.pad(pf_fr)
        pf_to_pad = self.pad(pf_to)
        qf_fr_pad = self.pad(qf_fr)
        qf_to_pad = self.pad(qf_to)

        pf_fr_bus = pf_fr_pad[:,self.bus_branchidxs_fr].sum(axis=2)
        # print('x0: ', pf_fr_bus.shape)
        pf_to_bus = pf_to_pad[:,self.bus_branchidxs_to].sum(axis=2)
        qf_fr_bus = qf_fr_pad[:,self.bus_branchidxs_fr].sum(axis=2)
        qf_to_bus = qf_to_pad[:,self.bus_branchidxs_to].sum(axis=2)

        # print(pf_fr.shape, pf_fr_pad.shape, pf_fr_bus.shape)

        return {"pf_fr":pf_fr, "pf_to":pf_to, "qf_fr":qf_fr, "qf_to":qf_to,
                "pf_fr_bus":pf_fr_bus, "pf_to_bus":pf_to_bus, "qf_fr_bus":qf_fr_bus, "qf_to_bus": qf_to_bus}
    
    def pad_ca(self, x):
        n, m = x.size()
        zeros_matrix = ca.DM.zeros(n, 1)  # Create a matrix of zeros with the same number of rows as x and one column
        return ca.horzcat(x, zeros_matrix)  # Vertically concatenate x and zeros_matrix along columns


    def compute_flow_ca(self, vm, dva, verbose=False):
        # print(self.bus_i, vm.shape)
        vmi = vm[:,self.bus_i]
        vmj = vm[:,self.bus_j]
        vmi2 = ca.power(vmi, 2)
        vmj2 = ca.power(vmj, 2)
        vmij = vmi*vmj

        vaij_cos = ca.cos(dva)
        vaij_sin = ca.sin(dva)

        # print(self.tap2.shape, self.br_g.shape, self.g_fr.shape, self.T_R.shape, self.T_I.shape)

        pf_fr = (1/ca.reshape(ca.DM(self.tap2), 1, -1)) * (ca.reshape(ca.DM(self.br_g), 1, -1) + ca.reshape(ca.DM(self.g_fr), 1, -1)) * vmi2\
                            + ((ca.reshape(ca.DM(-self.br_g), 1, -1) * ca.reshape(ca.DM(self.T_R), 1, -1) + ca.reshape(ca.DM(self.br_b), 1, -1) * ca.reshape(ca.DM(self.T_I), 1, -1))/ca.reshape(ca.DM(self.tap2), 1, -1)) * (vmij) * vaij_cos\
                            + ((ca.reshape(ca.DM(-self.br_b), 1, -1) * ca.reshape(ca.DM(self.T_R), 1, -1) - ca.reshape(ca.DM(self.br_g), 1, -1) * ca.reshape(ca.DM(self.T_I), 1, -1))/ca.reshape(ca.DM(self.tap2), 1, -1)) * (vmij) * vaij_sin
        # pf_fr = (1/(ca.DM(self.tap2))) * ((ca.DM(self.br_g)) + (ca.DM(self.g_fr))) * vmi2\
        #                     + (((ca.DM(-self.br_g)) * (ca.DM(self.T_R)) + (ca.DM(self.br_b)) * (ca.DM(self.T_I)))/(ca.DM(self.tap2))) * (vmij) * vaij_cos\
        #                     + (((ca.DM(-self.br_b)) * (ca.DM(self.T_R)) - (ca.DM(self.br_g)) * (ca.DM(self.T_I)))/(ca.DM(self.tap2))) * (vmij) * vaij_sin
        pf_to = (ca.reshape(ca.DM(self.br_g), 1, -1) + ca.reshape(ca.DM(self.g_to), 1, -1)) * vmj2\
                            + ((ca.reshape(ca.DM(-self.br_g), 1, -1) * ca.reshape(ca.DM(self.T_R), 1, -1) - ca.reshape(ca.DM(self.br_b), 1, -1) * ca.reshape(ca.DM(self.T_I), 1, -1))/ca.reshape(ca.DM(self.tap2), 1, -1)) * (vmij) * vaij_cos\
                            + ((ca.reshape(ca.DM(-self.br_b), 1, -1) * ca.reshape(ca.DM(self.T_R), 1, -1) + ca.reshape(ca.DM(self.br_g), 1, -1) * ca.reshape(ca.DM(self.T_I), 1, -1))/ca.reshape(ca.DM(self.tap2), 1, -1)) * (vmij) * (-vaij_sin)
        qf_fr = - (1/ca.reshape(ca.DM(self.tap2), 1, -1)) * (ca.reshape(ca.DM(self.br_b), 1, -1) + ca.reshape(ca.DM(self.b_fr), 1, -1)) * vmi2\
                            - ((ca.reshape(ca.DM(-self.br_b), 1, -1) * ca.reshape(ca.DM(self.T_R), 1, -1) - ca.reshape(ca.DM(self.br_g), 1, -1) * ca.reshape(ca.DM(self.T_I), 1, -1))/ca.reshape(ca.DM(self.tap2), 1, -1)) * (vmij) * vaij_cos\
                            + ((ca.reshape(ca.DM(-self.br_g), 1, -1) * ca.reshape(ca.DM(self.T_R), 1, -1) + ca.reshape(ca.DM(self.br_b), 1, -1) * ca.reshape(ca.DM(self.T_I), 1, -1))/ca.reshape(ca.DM(self.tap2), 1, -1)) * (vmij) * vaij_sin
        qf_to = -(ca.reshape(ca.DM(self.br_b), 1, -1) + ca.reshape(ca.DM(self.b_to), 1, -1)) * vmj2\
                            - ((ca.reshape(ca.DM(-self.br_b), 1, -1) * ca.reshape(ca.DM(self.T_R), 1, -1) + ca.reshape(ca.DM(self.br_g), 1, -1) * ca.reshape(ca.DM(self.T_I), 1, -1))/ca.reshape(ca.DM(self.tap2), 1, -1)) * (vmij) * vaij_cos\
                            + ((ca.reshape(ca.DM(-self.br_g), 1, -1) * ca.reshape(ca.DM(self.T_R), 1, -1) - ca.reshape(ca.DM(self.br_b), 1, -1) * ca.reshape(ca.DM(self.T_I), 1, -1))/ca.reshape(ca.DM(self.tap2), 1, -1)) * (vmij) * (-vaij_sin)
        # print('x1: ', pf_fr.shape)
        
        if verbose:
            print("compute_flow check params:: %.4f | %.4f | %.4f | %.4f"%(self.tap2.max(), self.br_g.max(), self.g_fr.max(), vmi2.max()),flush=True)
            term1 = (1/self.tap2) * (self.br_g + self.g_fr) * vmi2
            term2 = ((-self.br_g * self.T_R + self.br_b * self.T_I)/self.tap2) * (vmij) * vaij_cos
            term3 = ((-self.br_b * self.T_R - self.br_g * self.T_I)/self.tap2) * (vmij) * vaij_sin
            print("compute_flow:: %.4f | %.4f | %.4f "%(term1.max(), term2.max(), term3.max()),flush=True)

        pf_fr_pad = self.pad_ca(pf_fr)
        pf_to_pad = self.pad_ca(pf_to)
        qf_fr_pad = self.pad_ca(qf_fr)
        qf_to_pad = self.pad_ca(qf_to)

        # pf_fr_bus = pf_fr_pad[:,(ca.DM(self.bus_branchidxs_fr))].sum(axis=2)
        # pf_to_bus = pf_to_pad[:,(ca.DM(self.bus_branchidxs_to))].sum(axis=2)
        # qf_fr_bus = qf_fr_pad[:,(ca.DM(self.bus_branchidxs_fr))].sum(axis=2)
        # qf_to_bus = qf_to_pad[:,(ca.DM(self.bus_branchidxs_to))].sum(axis=2)

        # Sample size variables; replace with your actual dimensions
        num_samples, num_branches = self.bus_branchidxs_fr.shape

        # Assume bus_branchidxs_fr and bus_branchidxs_to are 2D arrays (Python lists)
        # containing the indices. Convert them to CasADi DM for compatibility.
        bus_branchidxs_fr_ca = ca.DM(self.bus_branchidxs_fr)
        bus_branchidxs_to_ca = ca.DM(self.bus_branchidxs_to)

        # Summing along the simulated third axis (which are rows in this 2D representation)
        pf_fr_bus_sum = []
        pf_to_bus_sum = []
        qf_fr_bus_sum = []
        qf_to_bus_sum = []

        for i in range(num_samples):
            pf_fr_block_sum = 0
            qf_fr_block_sum = 0
            for j in range(num_branches):
                pf_fr_block_sum += pf_fr_pad[:, int(bus_branchidxs_fr_ca[i, j])]
                qf_fr_block_sum += qf_fr_pad[:, int(bus_branchidxs_fr_ca[i, j])]
            
            pf_fr_bus_sum.append(pf_fr_block_sum)
            qf_fr_bus_sum.append(qf_fr_block_sum)

        # Sample size variables; replace with your actual dimensions
        num_samples, num_branches = self.bus_branchidxs_to.shape

        for i in range(num_samples):
            pf_to_block_sum = 0
            qf_to_block_sum = 0
            for j in range(num_branches):
                pf_to_block_sum += pf_to_pad[:, int(bus_branchidxs_to_ca[i, j])]
                qf_to_block_sum += qf_to_pad[:, int(bus_branchidxs_to_ca[i, j])]
            
            pf_to_bus_sum.append(pf_to_block_sum)
            qf_to_bus_sum.append(qf_to_block_sum)

        pf_fr_bus = ca.vertcat(*pf_fr_bus_sum).T
        pf_to_bus = ca.vertcat(*pf_to_bus_sum).T
        qf_fr_bus = ca.vertcat(*qf_fr_bus_sum).T
        qf_to_bus = ca.vertcat(*qf_to_bus_sum).T

        # print(pf_fr_bus.shape)


        return {"pf_fr":pf_fr, "pf_to":pf_to, "qf_fr":qf_fr, "qf_to":qf_to,
                "pf_fr_bus":pf_fr_bus, "pf_to_bus":pf_to_bus, "qf_fr_bus":qf_fr_bus, "qf_to_bus": qf_to_bus}
    

    def compute_torch_flow(self, vm, dva, verbose=False):

        torch_pad = torch.nn.ConstantPad2d((0,1,0,0),0.)

        vmi = vm[:,self.bus_i]
        vmj = vm[:,self.bus_j]
        vmi2 = vmi.pow(2)
        vmj2 = vmj.pow(2)
        vmij = vmi*vmj

        vaij_cos = torch.cos(dva)
        vaij_sin = torch.sin(dva)

        pf_fr = (1/torch.from_numpy(self.tap2)) * (torch.from_numpy(self.br_g) + torch.from_numpy(self.g_fr)) * vmi2\
                            + ((torch.from_numpy(-self.br_g) * torch.from_numpy(self.T_R) + torch.from_numpy(self.br_b) * torch.from_numpy(self.T_I))/torch.from_numpy(self.tap2)) * (vmij) * vaij_cos\
                            + ((torch.from_numpy(-self.br_b) * torch.from_numpy(self.T_R) - torch.from_numpy(self.br_g) * torch.from_numpy(self.T_I))/torch.from_numpy(self.tap2)) * (vmij) * vaij_sin
        pf_to = (torch.from_numpy(self.br_g) + torch.from_numpy(self.g_to)) * vmj2\
                            + ((torch.from_numpy(-self.br_g) * torch.from_numpy(self.T_R) - torch.from_numpy(self.br_b) * torch.from_numpy(self.T_I))/torch.from_numpy(self.tap2)) * (vmij) * vaij_cos\
                            + ((torch.from_numpy(-self.br_b) * torch.from_numpy(self.T_R) + torch.from_numpy(self.br_g) * torch.from_numpy(self.T_I))/torch.from_numpy(self.tap2)) * (vmij) * (-vaij_sin)
        qf_fr = - (1/torch.from_numpy(self.tap2)) * (torch.from_numpy(self.br_b) + torch.from_numpy(self.b_fr)) * vmi2\
                            - ((torch.from_numpy(-self.br_b) * torch.from_numpy(self.T_R) - torch.from_numpy(self.br_g) * torch.from_numpy(self.T_I))/torch.from_numpy(self.tap2)) * (vmij) * vaij_cos\
                            + ((torch.from_numpy(-self.br_g) * torch.from_numpy(self.T_R) + torch.from_numpy(self.br_b) * torch.from_numpy(self.T_I))/torch.from_numpy(self.tap2)) * (vmij) * vaij_sin
        qf_to = -(torch.from_numpy(self.br_b) + torch.from_numpy(self.b_to)) * vmj2\
                            - ((torch.from_numpy(-self.br_b) * torch.from_numpy(self.T_R) + torch.from_numpy(self.br_g) * torch.from_numpy(self.T_I))/torch.from_numpy(self.tap2)) * (vmij) * vaij_cos\
                            + ((torch.from_numpy(-self.br_g) * torch.from_numpy(self.T_R) - torch.from_numpy(self.br_b) * torch.from_numpy(self.T_I))/torch.from_numpy(self.tap2)) * (vmij) * (-vaij_sin)
        if verbose:
            print("compute_flow check params:: %.4f | %.4f | %.4f | %.4f"%(self.tap2.max(), self.br_g.max(), self.g_fr.max(), vmi2.max()),flush=True)
            term1 = (1/torch.from_numpy(self.tap2)) * (torch.from_numpy(self.br_g) + torch.from_numpy(self.g_fr)) * vmi2
            term2 = ((torch.from_numpy(-self.br_g) * torch.from_numpy(self.T_R) + torch.from_numpy(self.br_b) * torch.from_numpy(self.T_I))/torch.from_numpy(self.tap2)) * (vmij) * vaij_cos
            term3 = ((torch.from_numpy(-self.br_b) * torch.from_numpy(self.T_R) - torch.from_numpy(self.br_g) * torch.from_numpy(self.T_I))/torch.from_numpy(self.tap2)) * (vmij) * vaij_sin
            print("compute_flow:: %.4f | %.4f | %.4f "%(term1.max(), term2.max(), term3.max()),flush=True)

        pf_fr_pad = torch_pad(pf_fr)
        pf_to_pad = torch_pad(pf_to)
        qf_fr_pad = torch_pad(qf_fr)
        qf_to_pad = torch_pad(qf_to)

        # pf_fr_bus = pf_fr_pad[:,self.bus_branchidxs_fr].sum(dim=2)
        # pf_to_bus = pf_to_pad[:,self.bus_branchidxs_to].sum(dim=2)
        # qf_fr_bus = qf_fr_pad[:,self.bus_branchidxs_fr].sum(dim=2)
        # qf_to_bus = qf_to_pad[:,self.bus_branchidxs_to].sum(dim=2)

        # Summing along the simulated third axis (which are rows in this 2D representation)
        pf_fr_bus_sum = []
        pf_to_bus_sum = []
        qf_fr_bus_sum = []
        qf_to_bus_sum = []

        num_samples, num_branches = self.bus_branchidxs_fr.shape

        for i in range(num_samples):
            pf_fr_block_sum = 0
            qf_fr_block_sum = 0
            for j in range(num_branches):
                pf_fr_block_sum += pf_fr_pad[:, int(self.bus_branchidxs_fr[i, j])]
                qf_fr_block_sum += qf_fr_pad[:, int(self.bus_branchidxs_fr[i, j])]
            
            pf_fr_bus_sum.append(pf_fr_block_sum)
            qf_fr_bus_sum.append(qf_fr_block_sum)

        num_samples, num_branches = self.bus_branchidxs_to.shape

        for i in range(num_samples):
            pf_to_block_sum = 0
            qf_to_block_sum = 0
            for j in range(num_branches):
                pf_to_block_sum += pf_to_pad[:, int(self.bus_branchidxs_to[i, j])]
                qf_to_block_sum += qf_to_pad[:, int(self.bus_branchidxs_to[i, j])]
            
            pf_to_bus_sum.append(pf_to_block_sum)
            qf_to_bus_sum.append(qf_to_block_sum)

        pf_fr_bus = torch.cat(pf_fr_bus_sum).t()
        pf_to_bus = torch.cat(pf_to_bus_sum).t()
        qf_fr_bus = torch.cat(qf_fr_bus_sum).t()
        qf_to_bus = torch.cat(qf_to_bus_sum).t()

        return {"pf_fr":pf_fr, "pf_to":pf_to, "qf_fr":qf_fr, "qf_to":qf_to,
                "pf_fr_bus":pf_fr_bus, "pf_to_bus":pf_to_bus, "qf_fr_bus":qf_fr_bus, "qf_to_bus": qf_to_bus}


    def obj_fn(self, X, Y):
        pg = Y['pg']
        obj = self.quad_cost*pg**2 + self.lin_cost*pg + self.const_cost
        obj = obj.sum(axis=1) # sum over generators
        obj /= self.obj_scaler
        return obj.sum()
    
    def obj_fn_torch(self, X, pg, cost):
        # pg = Y['pg']#.to(self.device)
        obj = cost['quad_cost'].to(self.device)*pg.to(self.device)**2 + cost['lin_cost'].to(self.device)*pg.to(self.device) + cost['const_cost'].to(self.device)
        obj = obj.sum(dim=1) # sum over generators
        obj /= self.obj_scaler
        return obj

    def opt_gap(self, X, pg):
        obj_app = self.obj_fn_torch(X,Y)
        obj_gt = self.obj_fn_torch(X,Ygt)
        return torch.abs(obj_app-obj_gt) / torch.abs(obj_gt)
    
    def opt_gap_ca(self, X, Y, N, indices, cost):
        Y = ca.reshape(Y, 1, 208)
        Ypg = Y[:, indices[2]:indices[3]]
        quad = ca.repmat(ca.DM(cost['quad_cost']).T, 1, 1)
        lin = ca.repmat(ca.DM(cost['lin_cost']).T, 1, 1)
        const = ca.repmat(ca.DM(cost['const_cost']).T, 1, 1)

        obj = quad*Ypg**2 + lin*Ypg + const
        obj = ca.sum2(ca.sum1(obj)) # sum over generators
        obj /= self.obj_scaler
        return obj
    
    def opt_gap_alt(self, X, Y, indices, cost):
        Ypg = Y[:, indices[2]:indices[3]]
        obj_app = self.obj_fn_torch(X,Ypg, cost)
        return obj_app.sum()
    
    def ineq_resid(self, X ,Y):
        if "flow" in Y.keys():
            flow = Y["flow"]
            dva = Y["dva"]
        else:
            vm = Y["vm"]#.to(self.device)
            dva = Y["dva"]#.to(self.device)
            flow = self.compute_flow(vm,dva)

        # print(vm.shape, dva.shape)

        pf_fr = flow["pf_fr"]; qf_fr = flow["qf_fr"]
        pf_to = flow["pf_to"]; qf_to = flow["qf_to"]
        tl_fr = pf_fr**2 + qf_fr**2 - self.thermal_limit**2
        tl_to = pf_to**2 + qf_to**2 - self.thermal_limit**2

        ineq = np.concatenate([tl_fr,tl_to],axis=1)
        return ineq

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return np.maximum(resids, 0.)

    def eq_resid(self, X, Y):
        # power balance at each bus
        pd_bus = X['pd_bus']#.to(self.device)
        qd_bus = X['qd_bus']#.to(self.device)

        vm = Y["vm"]#.to(self.device)
        if "flow" in Y.keys():
            flow = Y["flow"]
            qg_bus = Y["qg_bus"]
            pg_bus = Y["pg_bus"]
        else:
            dva = Y["dva"]#.to(self.device)
            flow = self.compute_flow(vm,dva)
            pg = Y["pg"]#.to(self.device)
            qg = Y["qg"]#.to(self.device)

            # print(pg.shape)

            pg_pad = self.pad(pg)
            pg_bus = pg_pad[:,self.bus_genidxs].sum(axis=2)
            qg_pad = self.pad(qg)
            qg_bus = qg_pad[:,self.bus_genidxs].sum(axis=2)

        pf_fr_bus = flow["pf_fr_bus"]; pf_to_bus = flow["pf_to_bus"]
        qf_fr_bus = flow["qf_fr_bus"]; qf_to_bus = flow["qf_to_bus"]

        balance_p = pg_bus - pd_bus - pf_to_bus - pf_fr_bus - self.gs*vm**2
        balance_q = qg_bus - qd_bus - qf_to_bus - qf_fr_bus + self.bs*vm**2
        eq = np.concatenate([balance_p,balance_q])
        return eq


# JK
# Adapted from ACOPFProblem above
# TODO:
# REPLACE X features, keep Y the same
# Demand qd pd are neither X or Y
class ACOPFPredopt:
    def __init__(self, filepath, acopf_name, device, valid_frac=0.0833, test_frac=0.0833, obj_scaler=1e2):
        filepath = Path(filepath)
        self.acopf_name = acopf_name
        self.device = device
        self.valid_frac = valid_frac
        self.test_frac = test_frac
        self.train_frac = 1.-valid_frac-test_frac

        # laod network file
        network = json.load(open(filepath/"network.json"))
        self.nbus = len(network["bus"])
        self.ngen = len(network["gen"])
        self.nload = len(network["load"])
        self.nbranch = len(network["branch"])
        self.nshunt = len(network["shunt"])
        self.loadids = np.sort(np.array(list(network["load"].keys()),dtype=np.int64))
        self.genids = np.sort(np.array(list(network["gen"].keys()),dtype=np.int64))
        self.busids = np.sort(np.array(list(network["bus"].keys()),dtype=np.int64))
        self.branchids = np.sort(np.array(list(network["branch"].keys()),dtype=np.int64))
        self.shuntids = np.sort(np.array(list(network["shunt"].keys()),dtype=np.int64)) # note that id starts from 1 typically

        # look up slack bus
        self.slack_bus_idx = []
        for bus_id, bus_data in network["bus"].items():
            if bus_data['bus_type'] == 3:
                bus_idx = np.where(self.busids==int(bus_id))[0][0]
                self.slack_bus_idx.append(bus_idx)
        assert len(self.slack_bus_idx) == 1
        self.slack_bus_idx = self.slack_bus_idx[0]

        self.baseMVA = network["baseMVA"]
        self.obj_scaler = obj_scaler

        self.quad_cost, self.lin_cost, self.const_cost = torch.zeros(self.ngen,device=self.device),torch.zeros(self.ngen,device=self.device),torch.zeros(self.ngen,device=self.device)
        self.pgmin, self.pgmax, self.qgmin, self.qgmax = [], [], [], []
        self.gen2bus = [] # gen index to bus index

        # gen
        for i,id in enumerate(self.genids):
            gen_data = network["gen"][str(id)]
            cost_list = gen_data["cost"]
            for idx, c in enumerate(cost_list[::-1]):
                if idx == 0:
                    self.const_cost[i] = c
                elif idx == 1:
                    self.lin_cost[i] = c
                elif idx == 2:
                    self.quad_cost[i] = c
                else:
                    assert False
            self.pgmin.append(gen_data["pmin"]); self.qgmin.append(gen_data["qmin"]); self.pgmax.append(gen_data["pmax"]); self.qgmax.append(gen_data["qmax"])
            bus_id = gen_data["gen_bus"]
            bus_idx = np.where(self.busids==int(bus_id))[0][0]
            self.gen2bus.append(bus_idx)

        self.pgmin = torch.tensor(self.pgmin).to(self.device)
        self.pgmax = torch.tensor(self.pgmax).to(self.device)
        self.qgmin = torch.tensor(self.qgmin).to(self.device)
        self.qgmax = torch.tensor(self.qgmax).to(self.device)

        # bus
        self.vmmax, self.vmmin = [], []
        self.basekv = []
        for i,id in enumerate(self.busids):
            bus_data = network["bus"][str(id)]
            self.vmmax.append(bus_data["vmax"]); self.vmmin.append(bus_data["vmin"])
            self.basekv.append(bus_data["base_kv"])

        self.vmmax = torch.tensor(self.vmmax).to(self.device)
        self.vmmin = torch.tensor(self.vmmin).to(self.device)
        self.basekv = torch.tensor(self.basekv).to(self.device)

        # setup bus_genidxs
        bus_genidxs = []
        max_ngens = 0 # max num of generators at bus
        for i,id in enumerate(self.busids):
            genidxs = []
            for gen_idx, bus_idx in enumerate(self.gen2bus):
                if i==bus_idx:
                    genidxs.append(gen_idx)
            bus_genidxs.append(genidxs)
            max_ngens = max(max_ngens, len(genidxs))

        self.bus_genidxs = self.ngen*torch.ones(self.nbus,max_ngens, dtype=torch.int64).to(device)
        for i, genidxs in enumerate(bus_genidxs):
            for j,genidx in enumerate(genidxs):
                self.bus_genidxs[i,j] = genidx

        # load
        self.load2bus = [] # load idx to bus idx
        for i,id in enumerate(self.loadids):
            load_data = network["load"][str(id)]
            bus_id = load_data["load_bus"]
            bus_idx = np.where(self.busids==int(bus_id))[0][0]
            self.load2bus.append(bus_idx)

        # branch
        br_r, br_x = [], []
        tap, shift = [], [] # related to the transformer
        self.g_to, self.g_fr = [], []
        self.b_to, self.b_fr = [], []
        self.angmin, self.angmax = [], []
        self.bus_i, self.bus_j = [], [] # bus_i == f_bus | bus_j == t_bus || branch_out_per_bus==bus_i | branch_in_per_bus==bus_j #busidx (not id)
        self.thermal_limit = []
        self.edges = [] # i, j, and branch_id

        for i,id in enumerate(self.branchids):
            branch_data = network["branch"][str(id)]
            br_r.append(branch_data["br_r"]); br_x.append(branch_data['br_x'])
            tap.append(branch_data["tap"]); shift.append(branch_data["shift"])
            self.g_to.append(branch_data["g_to"]); self.g_fr.append(branch_data["g_fr"])
            self.b_to.append(branch_data["b_to"]); self.b_fr.append(branch_data["b_fr"])
            self.angmin.append(branch_data["angmin"]); self.angmax.append(branch_data["angmax"])
            self.thermal_limit.append(branch_data["rate_a"]) # only rate_a is considered in AC-OPF
            bus_i_id = branch_data["f_bus"]; bus_j_id = branch_data["t_bus"]
            bus_i_idx = np.where(self.busids==int(bus_i_id))[0][0]; bus_j_idx = np.where(self.busids==int(bus_j_id))[0][0]
            self.bus_i.append(bus_i_idx); self.bus_j.append(bus_j_idx)
            self.edges.append((self.bus_i[-1], self.bus_j[-1], {"idx": i}))

        br_r = torch.tensor(br_r).to(self.device)
        br_x = torch.tensor(br_x).to(self.device)
        tap = torch.tensor(tap).to(self.device)
        shift = torch.tensor(shift).to(self.device)
        self.g_to = torch.tensor(self.g_to).to(self.device)
        self.g_fr = torch.tensor(self.g_fr).to(self.device)
        self.b_to = torch.tensor(self.b_to).to(self.device)
        self.b_fr = torch.tensor(self.b_fr).to(self.device)
        self.angmin = torch.tensor(self.angmin).to(self.device) # radian
        self.angmax = torch.tensor(self.angmax).to(self.device) # radian
        self.bus_i = torch.tensor(self.bus_i).to(self.device)
        self.bus_j = torch.tensor(self.bus_j).to(self.device)
        self.thermal_limit = torch.tensor(self.thermal_limit).to(self.device)

        assert self.thermal_limit.max()>0.
        assert br_r.size() == br_x.size() and br_x.size() == self.g_to.size()
        assert self.g_to.size() == self.g_fr.size() and self.g_fr.size()[0] == self.nbranch

        br_r2_x2 = br_r.pow(2) + br_x.pow(2)
        self.br_g = br_r/br_r2_x2
        self.br_b = -br_x/br_r2_x2
        self.tap2 = tap.pow(2)
        self.T_R = tap * torch.cos(shift)
        self.T_I = tap * torch.sin(shift)

        # setup bus_branchidxs_fr, bus_branchidxs_to
        bus_branchidxs_fr, bus_branchidxs_to = [], []
        max_fr_nbranches, max_to_nbranches = 0, 0
        for i,id in enumerate(self.busids):
            branchidxs_fr, branchidxs_to = [], []
            for branch_idx, branch_bus_idx in enumerate(self.bus_i): # for from busidxs
                if i==branch_bus_idx:
                    branchidxs_fr.append(branch_idx)
            bus_branchidxs_fr.append(branchidxs_fr)
            max_fr_nbranches = max(max_fr_nbranches, len(branchidxs_fr))

            for branch_idx, branch_bus_idx in enumerate(self.bus_j): # for to busidxs
                if i==branch_bus_idx:
                    branchidxs_to.append(branch_idx)
            bus_branchidxs_to.append(branchidxs_to)
            max_to_nbranches = max(max_to_nbranches, len(branchidxs_to))

        self.bus_branchidxs_fr = self.nbranch*torch.ones(self.nbus,max_fr_nbranches,dtype=torch.int64).to(device)
        self.bus_branchidxs_to = self.nbranch*torch.ones(self.nbus,max_to_nbranches,dtype=torch.int64).to(device)
        for i,branchidxs_fr in enumerate(bus_branchidxs_fr):
            for j,branchidx in enumerate(branchidxs_fr):
                self.bus_branchidxs_fr[i,j] = branchidx
        for i,branchidxs_to in enumerate(bus_branchidxs_to):
            for j,branchidx in enumerate(branchidxs_to):
                self.bus_branchidxs_to[i,j] = branchidx

        # shunt
        self.gs = torch.zeros(self.nbus,dtype=torch.get_default_dtype()).to(device)
        self.bs = torch.zeros(self.nbus,dtype=torch.get_default_dtype()).to(device)

        for i,id in enumerate(self.shuntids):
            shunt_data = network["shunt"][str(id)]
            shunt_bus_id = shunt_data["shunt_bus"]
            shunt_bus_idx = np.where(self.busids==int(shunt_bus_id))[0][0]
            gs = shunt_data["gs"]
            bs = shunt_data["bs"]
            self.gs[shunt_bus_idx] += gs
            self.bs[shunt_bus_idx] += bs

        # data loading -- to cpu
        print(" Loading Data Instances...",flush=True)
        datapath = filepath/"data"
        datafiles = list(datapath.glob("*.json"))
        self.ndata = len(datafiles)
        self.va = torch.empty(self.ndata,self.nbus) # ground truth - voltage angle
        self.vm = torch.empty(self.ndata,self.nbus) # ground truth - voltage magnitude
        self.pg = torch.empty(self.ndata,self.ngen) # ground truth - active power generation
        self.qg = torch.empty(self.ndata,self.ngen) # ground truth - reactive power generation
        self.pd = torch.empty(self.ndata,self.nload) # input - active demand
        self.qd = torch.empty(self.ndata,self.nload) # input - reactive demand
        self.objs = torch.empty(self.ndata)
        # JK
        instance0 = json.load(open(datafiles[0],'r'))
        self.nfeat = len(instance0["feat"])
        self.feat = torch.empty(self.ndata,self.nfeat)


        for i, datafile in enumerate(datafiles):
            instance = json.load(open(datafile,'r'))
            self.va[i,:] = torch.tensor(instance["va"])   # voltage
            self.vm[i,:] = torch.tensor(instance["vm"])   # voltage
            self.pg[i,:] = torch.tensor(instance["pg"])   # active generation
            self.qg[i,:] = torch.tensor(instance["qg"])   # inactive generation
            self.pd[i,:] = torch.tensor(instance["pd"])   # active demand
            self.qd[i,:] = torch.tensor(instance["qd"])   # inactive demand
            self.objs[i] = instance["obj"] # just for reference
            # JK
            self.feat[i,:] = torch.tensor(instance["feat"]) # observed features (which map to demand)

        self.va = self.va
        self.vm = self.vm
        self.pg = self.pg
        self.qg = self.qg
        self.pd = self.pd
        self.qd = self.qd
        self.dva = self.va[:,self.bus_i] - self.va[:,self.bus_j]

        pd_extended = torch.zeros(self.ndata,self.nload,self.nbus,dtype=torch.get_default_dtype())
        qd_extended = torch.zeros(self.ndata,self.nload,self.nbus,dtype=torch.get_default_dtype())
        pd_extended[:,torch.arange(self.nload),self.load2bus] = self.pd
        qd_extended[:,torch.arange(self.nload),self.load2bus] = self.qd
        self.pd_bus = pd_extended.sum(dim=1)
        self.qd_bus = qd_extended.sum(dim=1)

        self.neq = 2*self.nbus # power balance (p and q)
        self.nineq = 2*self.nbranch

        self.pad = nn.ConstantPad2d((0,1,0,0),0.) # for padding flow to recover bus idx based tensor # used in 'compute_flow'

    def __str__(self):
        return self.acopf_name

    #TODO: Modify the training loop to avoid reaching for these old variables considered 'X'
    #      and get the features instead
    # Note: we could just add "feat" key to each of these, if sure the others will not be used as 'X'
    @property
    def X(self):
        #return {"feat":self.feat}
        return {"feat":self.feat,"pd":self.pd, "qd":self.qd, "pd_bus":self.pd_bus, "qd_bus":self.qd_bus}

    @property
    def trainX(self):
        #return {"feat":self.feat[:int(self.ndata * self.train_frac)]}
        return {"feat":self.feat[:int(self.ndata * self.train_frac)], "pd":self.pd[:int(self.ndata * self.train_frac)], "qd":self.qd[:int(self.ndata * self.train_frac)],
                "pd_bus":self.pd_bus[:int(self.ndata * self.train_frac)], "qd_bus":self.qd_bus[:int(self.ndata * self.train_frac)]}

    @property
    def validX(self):
        #return {"feat":self.feat[int(self.ndata * self.train_frac):int(self.ndata * (self.train_frac + self.valid_frac))]}
        return {"feat":self.feat[int(self.ndata * self.train_frac):int(self.ndata * (self.train_frac + self.valid_frac))],
                "pd":self.pd[int(self.ndata * self.train_frac):int(self.ndata * (self.train_frac + self.valid_frac))],
                "qd":self.qd[int(self.ndata * self.train_frac):int(self.ndata * (self.train_frac + self.valid_frac))],
                "pd_bus":self.pd_bus[int(self.ndata * self.train_frac):int(self.ndata * (self.train_frac + self.valid_frac))],
                "qd_bus":self.qd_bus[int(self.ndata * self.train_frac):int(self.ndata * (self.train_frac + self.valid_frac))]}

    @property
    def testX(self):
        #return {"feat":self.feat[int(self.ndata * (self.train_frac + self.valid_frac)):]}
        return {"feat":self.feat[int(self.ndata * (self.train_frac + self.valid_frac)):],
                "pd":self.pd[int(self.ndata * (self.train_frac + self.valid_frac)):],
                "qd":self.qd[int(self.ndata * (self.train_frac + self.valid_frac)):],
                "pd_bus":self.pd_bus[int(self.ndata * (self.train_frac + self.valid_frac)):],
                "qd_bus":self.qd_bus[int(self.ndata * (self.train_frac + self.valid_frac)):]}

    @property
    def Y(self):
        return {"va":self.va, "vm":self.vm, "pg":self.pg, "qg":self.qg, "dva": self.dva}

    @property
    def trainY(self):
        return {"va":self.va[:int(self.ndata*self.train_frac)],
                "vm":self.vm[:int(self.ndata*self.train_frac)],
                "pg":self.pg[:int(self.ndata*self.train_frac)],
                "qg":self.qg[:int(self.ndata*self.train_frac)],
                "dva":self.dva[:int(self.ndata*self.train_frac)]
                }

    @property
    def validY(self):
        return {"va":self.va[int(self.ndata*self.train_frac):int(self.ndata*(self.train_frac + self.valid_frac))],
                "vm":self.vm[int(self.ndata*self.train_frac):int(self.ndata*(self.train_frac + self.valid_frac))],
                "pg":self.pg[int(self.ndata*self.train_frac):int(self.ndata*(self.train_frac + self.valid_frac))],
                "qg":self.qg[int(self.ndata*self.train_frac):int(self.ndata*(self.train_frac + self.valid_frac))],
                "dva":self.dva[int(self.ndata*self.train_frac):int(self.ndata*(self.train_frac + self.valid_frac))]
                }

    @property
    def testY(self):
        return {"va":self.va[int(self.ndata*(self.train_frac + self.valid_frac)):],
                "vm":self.vm[int(self.ndata*(self.train_frac + self.valid_frac)):],
                "pg":self.pg[int(self.ndata*(self.train_frac + self.valid_frac)):],
                "qg":self.qg[int(self.ndata*(self.train_frac + self.valid_frac)):],
                "dva":self.dva[int(self.ndata*(self.train_frac + self.valid_frac)):]
                }

    def compute_flow(self, vm, dva, verbose=False):
        vmi = vm[:,self.bus_i]
        vmj = vm[:,self.bus_j]
        vmi2 = vmi.pow(2)
        vmj2 = vmj.pow(2)
        vmij = vmi*vmj

        vaij_cos = torch.cos(dva)
        vaij_sin = torch.sin(dva)

        pf_fr = (1/self.tap2) * (self.br_g + self.g_fr) * vmi2\
                            + ((-self.br_g * self.T_R + self.br_b * self.T_I)/self.tap2) * (vmij) * vaij_cos\
                            + ((-self.br_b * self.T_R - self.br_g * self.T_I)/self.tap2) * (vmij) * vaij_sin
        pf_to = (self.br_g + self.g_to) * vmj2\
                            + ((-self.br_g * self.T_R - self.br_b * self.T_I)/self.tap2) * (vmij) * vaij_cos\
                            + ((-self.br_b * self.T_R + self.br_g * self.T_I)/self.tap2) * (vmij) * (-vaij_sin)
        qf_fr = - (1/self.tap2) * (self.br_b + self.b_fr) * vmi2\
                            - ((-self.br_b * self.T_R - self.br_g * self.T_I)/self.tap2) * (vmij) * vaij_cos\
                            + ((-self.br_g * self.T_R + self.br_b * self.T_I)/self.tap2) * (vmij) * vaij_sin
        qf_to = -(self.br_b + self.b_to) * vmj2\
                            - ((-self.br_b * self.T_R + self.br_g * self.T_I)/self.tap2) * (vmij) * vaij_cos\
                            + ((-self.br_g * self.T_R - self.br_b * self.T_I)/self.tap2) * (vmij) * (-vaij_sin)
        if verbose:
            print("compute_flow check params:: %.4f | %.4f | %.4f | %.4f"%(self.tap2.max(), self.br_g.max(), self.g_fr.max(), vmi2.max()),flush=True)
            term1 = (1/self.tap2) * (self.br_g + self.g_fr) * vmi2
            term2 = ((-self.br_g * self.T_R + self.br_b * self.T_I)/self.tap2) * (vmij) * vaij_cos
            term3 = ((-self.br_b * self.T_R - self.br_g * self.T_I)/self.tap2) * (vmij) * vaij_sin
            print("compute_flow:: %.4f | %.4f | %.4f "%(term1.max(), term2.max(), term3.max()),flush=True)

        pf_fr_pad = self.pad(pf_fr)
        pf_to_pad = self.pad(pf_to)
        qf_fr_pad = self.pad(qf_fr)
        qf_to_pad = self.pad(qf_to)

        pf_fr_bus = pf_fr_pad[:,self.bus_branchidxs_fr].sum(dim=2)
        pf_to_bus = pf_to_pad[:,self.bus_branchidxs_to].sum(dim=2)
        qf_fr_bus = qf_fr_pad[:,self.bus_branchidxs_fr].sum(dim=2)
        qf_to_bus = qf_to_pad[:,self.bus_branchidxs_to].sum(dim=2)

        return {"pf_fr":pf_fr, "pf_to":pf_to, "qf_fr":qf_fr, "qf_to":qf_to,
                "pf_fr_bus":pf_fr_bus, "pf_to_bus":pf_to_bus, "qf_fr_bus":qf_fr_bus, "qf_to_bus": qf_to_bus}

    def obj_fn(self, X, Y):
        pg = Y['pg'].to(self.device)
        obj = self.quad_cost*pg**2 + self.lin_cost*pg + self.const_cost
        obj = obj.sum(dim=1) # sum over generators
        obj /= self.obj_scaler
        return obj

    def opt_gap(self, X, Y, Ygt):
        obj_app = self.obj_fn(X,Y)
        obj_gt = self.obj_fn(X,Ygt)
        return (obj_app-obj_gt).abs()/obj_gt.abs()

    def ineq_resid(self, X ,Y):
        if "flow" in Y.keys():
            flow = Y["flow"]
            dva = Y["dva"]
        else:
            vm = Y["vm"].to(self.device)
            dva = Y["dva"].to(self.device)
            flow = self.compute_flow(vm,dva)

        pf_fr = flow["pf_fr"]; qf_fr = flow["qf_fr"]
        pf_to = flow["pf_to"]; qf_to = flow["qf_to"]
        tl_fr = pf_fr**2 + qf_fr**2 - self.thermal_limit**2
        tl_to = pf_to**2 + qf_to**2 - self.thermal_limit**2

        ineq = torch.cat([tl_fr,tl_to],dim=1)
        return ineq

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0.)

    def eq_resid(self, X, Y):
        # power balance at each bus
        pd_bus = X['pd_bus'].to(self.device)
        qd_bus = X['qd_bus'].to(self.device)

        vm = Y["vm"].to(self.device)
        if "flow" in Y.keys():
            flow = Y["flow"]
            qg_bus = Y["qg_bus"]
            pg_bus = Y["pg_bus"]
        else:
            dva = Y["dva"].to(self.device)
            flow = self.compute_flow(vm,dva)
            pg = Y["pg"].to(self.device)
            qg = Y["qg"].to(self.device)

            pg_pad = self.pad(pg)
            pg_bus = pg_pad[:,self.bus_genidxs].sum(dim=2)
            qg_pad = self.pad(qg)
            qg_bus = qg_pad[:,self.bus_genidxs].sum(dim=2)

        pf_fr_bus = flow["pf_fr_bus"]; pf_to_bus = flow["pf_to_bus"]
        qf_fr_bus = flow["qf_fr_bus"]; qf_to_bus = flow["qf_to_bus"]
        balance_p = pg_bus - pd_bus - pf_to_bus - pf_fr_bus - self.gs*vm**2
        balance_q = qg_bus - qd_bus - qf_to_bus - qf_fr_bus + self.bs*vm**2
        eq = torch.cat([balance_p,balance_q], dim=1)
        return eq

###################################################################
# NEURAL NETWORKS
###################################################################

def init_layer(nlayer, nhidden, nin, nout, primal=True):
    layer_sizes = [nin]
    layer_sizes += nlayer*[nhidden]
    layers = reduce(operator.add, [[nn.Linear(a,b), nn.ReLU()] for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
    if primal:
        layers += [nn.Linear(layer_sizes[-1],nout), nn.ReLU(), nn.Linear(nout,nout), nn.Hardsigmoid()]
        return layers
    else:
        return layers+[nn.Linear(layer_sizes[-1],nout)]

class NNPrimalACOPFSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.data = data
        self.device = data.device
        self.npg = data.ngen
        self.nqg = data.ngen
        self.nvm = data.nbus
        self.nva = data.nbus-1
        self.nbranch = data.nbranch
        self.ndva = self.nbranch
        self.xdim = 2*data.nload # pd and qd
        self.ydim = self.npg+self.nqg+self.nvm+self.ndva
        self.nonslack_busidxs = torch.arange(data.nbus)
        self.nonslack_busidxs = self.nonslack_busidxs[self.nonslack_busidxs!=data.slack_bus_idx]
        print("X dim:%d, Y dim:%d"%(self.xdim, self.ydim),flush=True)

        self.pgmin = data.pgmin; self.pgmax = data.pgmax
        self.qgmin = data.qgmin; self.qgmax = data.qgmax
        self.vmmin = data.vmmin; self.vmmax = data.vmmax
        self.dvamin = data.angmin; self.dvamax = data.angmax

        fraction = args['hiddenfrac']

        # combinded net
        nlayer = args['nlayer']
        nhidden = int(fraction*self.ydim)
        layer_sizes = [self.xdim]
        layer_sizes += nlayer*[nhidden]
        layers = reduce(operator.add, [[nn.Linear(a,b), nn.ReLU()] for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        self.net = nn.Sequential(*layers)
        self.pg = nn.Sequential(nn.Linear(layer_sizes[-1],self.npg), nn.ReLU(), nn.Linear(self.npg,self.npg), nn.Hardsigmoid())
        self.qg = nn.Sequential(nn.Linear(layer_sizes[-1],self.nqg), nn.ReLU(), nn.Linear(self.nqg,self.nqg), nn.Hardsigmoid())
        self.vm = nn.Sequential(nn.Linear(layer_sizes[-1],self.nvm), nn.ReLU(), nn.Linear(self.nvm,self.nvm), nn.Hardsigmoid())
        self.dva = nn.Sequential(nn.Linear(layer_sizes[-1],self.ndva), nn.ReLU(), nn.Linear(self.ndva,self.ndva), nn.Hardsigmoid())

    def forward(self, x):
        pd = x["pd"]; qd = x["qd"]
        assert pd.shape[0] == qd.shape[0]

        pd = pd.to(self.device); qd = qd.to(self.device)
        x = torch.cat([pd,qd],dim=1)
        # combined net
        out = self.net(x)
        pg = self.pg(out)
        qg = self.qg(out)
        vm = self.vm(out)
        dva = self.dva(out)

        pg = (self.pgmax-self.pgmin)*pg + self.pgmin
        qg = (self.qgmax-self.qgmin)*qg + self.qgmin

        pg_pad = self.data.pad(pg)
        qg_pad = self.data.pad(qg)
        pg_bus = pg_pad[:,self.data.bus_genidxs].sum(dim=2)
        qg_bus = qg_pad[:,self.data.bus_genidxs].sum(dim=2)

        vm = (self.vmmax-self.vmmin)*vm + self.vmmin
        dva = (self.dvamax-self.dvamin)*dva + self.dvamin
        flow = self.data.compute_flow(vm, dva) # flow -- Ohm's law

        return {"pg": pg, "qg": qg, "pg_bus": pg_bus, "qg_bus": qg_bus,
                "vm": vm, "dva": dva, "flow":flow}


class NNDualACOPFSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.device = data.device
        self.xdim = 2*data.nload # pd and qd

        fraction = args['hiddenfrac']
        # combinded net
        nout = data.neq+data.nineq
        nhidden = int(fraction*nout)
        nlayer = args['nlayer']
        layer_sizes = [self.xdim]
        layer_sizes += nlayer*[nhidden]
        layers = reduce(operator.add, [[nn.Linear(a,b), nn.ReLU()] for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        self.net = nn.Sequential(*layers)
        self.lag_eq = nn.Sequential(nn.Linear(layer_sizes[-1],data.neq), nn.ReLU(), nn.Linear(data.neq,data.neq))
        self.lag_ineq = nn.Sequential(nn.Linear(layer_sizes[-1],data.nineq), nn.ReLU(), nn.Linear(data.nineq,data.nineq))

        nn.init.zeros_(self.lag_eq[-1].weight); nn.init.zeros_(self.lag_eq[-1].bias);
        nn.init.zeros_(self.lag_ineq[-1].weight); nn.init.zeros_(self.lag_ineq[-1].bias);


    def forward(self, x):
        pd = x["pd"]; qd = x["qd"]
        pd = pd.to(self.device); qd = qd.to(self.device)

        # x = torch.cat([pd,qd],dim=1) #!
        x = torch.cat([pd,qd],dim=0)

        # combined net
        out = self.net(x)
        return self.lag_eq(out), self.lag_ineq(out)


# JK
# Adapted from NNPrimalACOPFSolver
class NNPredoptACOPFSolver(nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.data = data
        self.device = data.device
        self.npg = data.ngen
        self.nqg = data.ngen
        self.nvm = data.nbus
        self.nva = data.nbus-1
        self.nbranch = data.nbranch
        self.ndva = self.nbranch
        self.xdim = 2*data.nload # pd and qd
        self.ydim = self.npg+self.nqg+self.nvm+self.ndva
        self.nonslack_busidxs = torch.arange(data.nbus)
        self.nonslack_busidxs = self.nonslack_busidxs[self.nonslack_busidxs!=data.slack_bus_idx]
        print("X dim:%d, Y dim:%d"%(self.xdim, self.ydim),flush=True)

        self.pgmin = data.pgmin; self.pgmax = data.pgmax
        self.qgmin = data.qgmin; self.qgmax = data.qgmax
        self.vmmin = data.vmmin; self.vmmax = data.vmmax
        self.dvamin = data.angmin; self.dvamax = data.angmax

        # JK
        self.nfeat = data.nfeat
        self.npd = data.nload
        self.nqd = data.nload

        fraction = args['hiddenfrac']

        # combinded net
        nlayer = args['nlayer']
        nhidden = int(fraction*self.ydim)

        layer_sizes = [self.xdim]
        layer_sizes += nlayer*[nhidden]

        layers = reduce(operator.add, [[nn.Linear(a,b), nn.ReLU()] for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])

        self.net = nn.Sequential(*layers)
        self.pg = nn.Sequential(nn.Linear(layer_sizes[-1],self.npg), nn.ReLU(), nn.Linear(self.npg,self.npg), nn.Hardsigmoid())
        self.qg = nn.Sequential(nn.Linear(layer_sizes[-1],self.nqg), nn.ReLU(), nn.Linear(self.nqg,self.nqg), nn.Hardsigmoid())
        self.vm = nn.Sequential(nn.Linear(layer_sizes[-1],self.nvm), nn.ReLU(), nn.Linear(self.nvm,self.nvm), nn.Hardsigmoid())
        self.dva = nn.Sequential(nn.Linear(layer_sizes[-1],self.ndva), nn.ReLU(), nn.Linear(self.ndva,self.ndva), nn.Hardsigmoid())
        # JK
        # should we use hard sigmoid e.g. to restrict the output range?
        # add more layers depending on complexity of feat -> pd and feat -> qd
        self.feat2pd = nn.Sequential(nn.Linear(self.nfeat,self.npd), nn.ReLU(), nn.Linear(self.npd,self.npd), nn.ReLU())
        self.feat2qd = nn.Sequential(nn.Linear(self.nfeat,self.nqd), nn.ReLU(), nn.Linear(self.nqd,self.nqd), nn.ReLU())



    def forward(self, x):
        #pd = x["pd"]; qd = x["qd"]
        feat = x["feat"]
        feat = feat.to(self.device)
        pd = self.feat2pd(feat); qd = self.feat2qd(feat)
        assert pd.shape[0] == qd.shape[0]

        #pd = pd.to(self.device); qd = qd.to(self.device)
        x = torch.cat([pd,qd],dim=1)
        # combined net
        out = self.net(x)
        pg = self.pg(out)
        qg = self.qg(out)
        vm = self.vm(out)
        dva = self.dva(out)

        pg = (self.pgmax-self.pgmin)*pg + self.pgmin
        qg = (self.qgmax-self.qgmin)*qg + self.qgmin

        pg_pad = self.data.pad(pg)
        qg_pad = self.data.pad(qg)
        pg_bus = pg_pad[:,self.data.bus_genidxs].sum(dim=2)
        qg_bus = qg_pad[:,self.data.bus_genidxs].sum(dim=2)

        vm = (self.vmmax-self.vmmin)*vm + self.vmmin
        dva = (self.dvamax-self.dvamin)*dva + self.dvamin
        flow = self.data.compute_flow(vm, dva) # flow -- Ohm's law

        return {"pg": pg, "qg": qg, "pg_bus": pg_bus, "qg_bus": qg_bus,
                "vm": vm, "dva": dva, "flow":flow}



def load_acopf_data(args, current_path, device):
    datapath = current_path/"datasets"/"acopf"

    filesubpaths = {
        "acopf57":"pglib_opf_case57_ieee",
        "acopf118":"pglib_opf_case118_ieee",
        "acopf57sad":"pglib_opf_case57_ieee__sad",
        "acopf118sad":"pglib_opf_case118_ieee__sad",
    }
    filepath = datapath/filesubpaths[args['probtype']]
    obj_scaler = args['objscaler'] if 'objscaler' in args.keys() and args['objscaler'] is not None else 1e5
    data = ACOPFProblem(filepath, args['probtype'], device)
    args['nex'] = data.ndata
    args['nineq'] = data.nineq
    args['neq'] = data.neq
    print("Problem %s redefines the configuration--> nex:%d | nineq:%d | neq:%d"%(args['probtype'],data.ndata,data.nineq,data.neq),flush=True)
    print("   #bus:%d | #gen:%d | #load:%d | #branch:%d"%(data.nbus, data.ngen, data.nload, data.nbranch),flush=True)
    return data, args


# JK
def load_acopf_predopt_data(args, current_path, device):
    datapath = current_path/"datasets"/"acopf"

    filesubpaths = {
        "predopt_acopf57":"pglib_opf_predopt_case57_ieee",
        "predopt_acopf118":"pglib_opf_predopt_case118_ieee",
        "predopt_acopf57sad":"pglib_opf_predopt_case57_ieee__sad",
        "predopt_acopf118sad":"pglib_opf_predopt_case118_ieee__sad",
    }
    filepath = datapath/filesubpaths[args['probtype']]
    obj_scaler = args['objscaler'] if 'objscaler' in args.keys() and args['objscaler'] is not None else 1e5
    data = ACOPFPredopt(filepath, args['probtype'], device, obj_scaler=obj_scaler)
    args['nex'] = data.ndata
    args['nineq'] = data.nineq
    args['neq'] = data.neq
    print("Problem %s redefines the configuration--> nex:%d | nineq:%d | neq:%d"%(args['probtype'],data.ndata,data.nineq,data.neq),flush=True)
    print("   #bus:%d | #gen:%d | #load:%d | #branch:%d"%(data.nbus, data.ngen, data.nload, data.nbranch),flush=True)
    return data, args

# JK
# TODO: Maybe this file should go in datasets/ for coherence (if so current_path etc below need change)
# featmap is a function that maps [pd, qd] to underlying observable features
# current_path is the overall directory of this project
# target_dir is the (yet nonexistent) directory where our new data will go
# source_dir is the directory where we get our base ac_opf data instances
# This code appends feature data to a copy of each of those instances based on
#     and stores it in target_dir

def convert_acopf_predopt_data(probtype, current_path, featmap):
    datapath = current_path/"datasets"/"acopf"


    filesubpaths_source = {
        "acopf57":"pglib_opf_case57_ieee",
        "acopf118":"pglib_opf_case118_ieee",
        "acopf57sad":"pglib_opf_case57_ieee__sad",
        "acopf118sad":"pglib_opf_case118_ieee__sad",
    }

    filesubpaths_target = {
        "acopf57":"pglib_opf_predopt_case57_ieee",
        "acopf118":"pglib_opf_predopt_case118_ieee",
        "acopf57sad":"pglib_opf_predopt_case57_ieee__sad",
        "acopf118sad":"pglib_opf_predopt_case118_ieee__sad",
    }
    sourcepath = datapath/filesubpaths_source[probtype]
    targetpath = datapath/filesubpaths_target[probtype]

    # data loading -- to cpu
    print(" Loading Data Instances...",flush=True)
    datapath = sourcepath/"data"
    source_datafiles = list(datapath.glob("*.json"))
    for i, sourcefile in enumerate(source_datafiles):
        instance = json.load(open(sourcefile,'r'))
        pd = torch.Tensor(instance["pd"])
        qd = torch.Tensor(instance["qd"])
        pd_qd = torch.cat((pd,qd))
        instance["feat"] = featmap(pd_qd).tolist()
        with open(os.path.join(targetpath,"data",str(int(1e11*pd[0]))+".json"), "w") as fp:
            json.dump(instance , fp)



class PDLDataSet(Dataset):
    def __init__(self, X):
        super().__init__()
        self.X = X
        try:
            self.nex = self.X.shape[0]
        except:
            self.nex = self.X["pd"].shape[0]

    def __len__(self):
        return self.nex

    def __getitem__(self, idx):
        if isinstance(self.X,dict):
            return {k:v[idx] for k,v in self.X.items()}
        else:
            return self.X[idx]
        