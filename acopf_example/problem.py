import torch
import casadi as ca


class problem(object):
    def __init__(self, x, x_dict, y, y_dict, acopf_problem, yindices, xindices, M_eq, M_in, cost, verbose=True):
        self.ngen = acopf_problem.ngen
        self.obj_scaler = 1e5
        self.x = x
        self.x_dict = x_dict
        self.y = y
        self.y_dict = y_dict
        self.acopf_problem = acopf_problem
        self.yindices = yindices
        self.xindices = xindices

        self.M_eq = M_eq
        self.M_in = M_in

        self.torch_pad = torch.nn.ConstantPad2d((0,1,0,0),0.)


        self.verbose = verbose
        self.dual = None
        self.cost = cost


    def constraints_ca(self, y, N):

        y = ca.reshape(y, 1, 208)

        ## Extract elements from y ##
        pd_bus = ca.reshape(ca.DM(self.x[:, self.xindices[3]:self.xindices[4]].tolist()), 1, self.x_dict['pd_bus'].shape[1])
        qd_bus = ca.reshape(ca.DM(self.x[:, self.xindices[4]:self.xindices[5]].tolist()), 1, self.x_dict['qd_bus'].shape[1])

        vm = ca.reshape(y[:, self.yindices[1]:self.yindices[2]], (1,57))
        dva = ca.reshape(y[:, self.yindices[-2]:self.yindices[-1]], (1,80))


        ## Compute flow ##
        pf_fr = []
        pf_to = []
        qf_fr = []
        qf_to = []
        pf_fr_bus = []
        pf_to_bus = []
        qf_fr_bus = []
        qf_to_bus = []

        for _ in range(1):
            flow_tmp = self.acopf_problem.compute_flow_ca(vm[:,:], dva[:,:])
            pf_fr.append(flow_tmp['pf_fr'])
            pf_to.append(flow_tmp['pf_to'])
            qf_fr.append(flow_tmp['qf_fr'])
            qf_to.append(flow_tmp['qf_to'])
            pf_fr_bus.append(flow_tmp['pf_fr_bus'])
            pf_to_bus.append(flow_tmp['pf_to_bus'])
            qf_fr_bus.append(flow_tmp['qf_fr_bus'])
            qf_to_bus.append(flow_tmp['qf_to_bus'])


        flow = {
            "pf_fr": ca.vertcat(*pf_fr), 
            "pf_to": ca.vertcat(*pf_to), 
            "qf_fr": ca.vertcat(*qf_fr), 
            "qf_to": ca.vertcat(*qf_to),
            "pf_fr_bus": ca.vertcat(*pf_fr_bus), 
            "pf_to_bus": ca.vertcat(*pf_to_bus),
            "qf_fr_bus": ca.vertcat(*qf_fr_bus), 
            "qf_to_bus": ca.vertcat(*qf_to_bus)
        }


        pg = ca.reshape(y[:, self.yindices[2]:self.yindices[3]], (1,(self.yindices[3]-self.yindices[2])))
        qg = ca.reshape(y[:, self.yindices[3]:self.yindices[4]], (1,(self.yindices[4]-self.yindices[3])))

        pg_pad = self.acopf_problem.pad_ca(pg)
        pg_bus = pg_pad[:,self.acopf_problem.bus_genidxs]
        qg_pad = self.acopf_problem.pad_ca(qg)
        qg_bus = qg_pad[:,self.acopf_problem.bus_genidxs]

        ## Calculate eq_resid ##

        pf_fr_bus = flow["pf_fr_bus"]; pf_to_bus = flow["pf_to_bus"]
        qf_fr_bus = flow["qf_fr_bus"]; qf_to_bus = flow["qf_to_bus"]

        gs_replicated = ca.repmat(ca.DM(self.acopf_problem.gs).T, 1, 1)
        bs_replicated = ca.repmat(ca.DM(self.acopf_problem.bs).T, 1, 1)
    
        balance_p = pg_bus - pd_bus - pf_to_bus - pf_fr_bus - gs_replicated*vm**2
        balance_q = qg_bus - qd_bus - qf_to_bus - qf_fr_bus + bs_replicated*vm**2
        eq = ca.fabs(ca.vertcat(*[balance_p,balance_q]))


        ## Calculate ineq_resid ##

        pf_fr = flow["pf_fr"]; qf_fr = flow["qf_fr"]
        pf_to = flow["pf_to"]; qf_to = flow["qf_to"]
        thermal_limit_replicated = ca.repmat(ca.DM(self.acopf_problem.thermal_limit).T, 1, 1)
        tl_fr = pf_fr**2 + qf_fr**2 - thermal_limit_replicated**2
        tl_to = pf_to**2 + qf_to**2 - thermal_limit_replicated**2
        ineq = ca.vertcat(*[tl_fr,tl_to])
        ineq = ca.fmax(ineq, 0.)

        residuals =  ca.reshape(ca.horzcat(*[eq, ineq]), -1, 1)

        return residuals



def reconstruct_dict(y, y_dict, indices):

    dict_obj = {}
    i = 0
    for item in y_dict.keys():
        dict_obj[item] = y[:, indices[i]:indices[i+1]]
        i += 1
    
    return dict_obj
