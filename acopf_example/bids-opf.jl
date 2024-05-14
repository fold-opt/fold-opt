using Random, Distributions
using JSON
using ArgParse
using JuMP, Ipopt, PowerModels
PowerModels.silence() # suppress warning and info messages
#PowerModels.silence() # suppress warning and info messages
Random.seed!(123)

""" Parse Arguments """
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--netname", "-n"
            help = "The input network name"
            arg_type = String
            default = "pglib_opf_case57_ieee"
        "--output", "-o"
            help = "the output name"
            arg_type = String
            default = "traindata_ext"
        "--lb"
            help = "The lb (in %) of the load interval"
            arg_type = Float64
            default = 0.8
        "--ub"
            help = "The ub (in %) of the load interval"
            arg_type = Float64
            default = 1.2
        "--step"
            help = "The step size resulting in a new load x + step"
            arg_type = Float64
            default = 0.1
        "--nperm"
            help = "The number of load permutations for each laod scale"
            arg_type = Int
            default = 10
    end
    return parse_args(s)
end

args = parse_commandline()

## Load data
#data_path = "data/"
data_path = "/home/jacob/Documents/acopf/datasets/acopf/pglib-opf"
#outdir = data_path * "/traindata/" * args["netname"]
#outdir = "datasets/acopf/ " * args["netname"] * "/traindata/"
outdir = "/home/jacob/Documents/acopf/datasets/acopf/pglib_opf_case57_ieee/data/"
#fileout = outdir * "/" * args["output"]  * ".json"

mkpath(outdir)
filein = data_path * "/" * args["netname"] * ".m"
data = PowerModels.parse_file(filein)
Load_range = collect(args["lb"]:args["step"]:(args["ub"]))
solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer) #, print_level=0)
nloads = length(data["load"])
res_stack = []

#--- Data features ---#

# weather generation [heat] (next day - correlated)
_maxT = 110
_minT = 20
_data_size = 64
_prob_bids = 0.4

prev_day_T = rand(Uniform(_minT, _maxT),_data_size)

next_day_T = prev_day_T .+ rand(Normal(0, 20), _data_size)
ΔT = abs.(next_day_T - prev_day_T)

# generator cost generation (proportional to heat) c1, c2
gen_cost = Dict(name => gen["cost"] for (name, gen) in data["gen"]
                                if data["gen"][name]["pmax"] > 0)

# Add random perturbations
for name in keys(gen_cost)
    gen_cost[name][1] += rand(Normal(0, gen_cost[name][1]/100))
    gen_cost[name][2] += rand(Normal(0, gen_cost[name][2]/100))
end

# increase cost as a percentage of the ΔT
gen_cost_data = Dict( name => (gen_cost[name][1] .+ gen_cost[name][1] .* (ΔT / 100),
                    gen_cost[name][2] .+ gen_cost[name][2] .* (ΔT / 100))
                    for name in keys(gen_cost))

# unweighted cost
unweighted_cost_data = Dict( name => (gen_cost[name][1] .+ gen_cost[name][1],
                        gen_cost[name][2] .+ gen_cost[name][2])
                        for name in keys(gen_cost))

# check how far is this from an ideal temperature
# the higher it is the more extra-load we are going to require.
extra_load = abs.((next_day_T .- 65) / 45)
# normalize the above into [0, 0.2] to use as multiplier
extra_load_mult = extra_load / maximum(extra_load) * 0.2

load_bids_data = Dict(name => Dict("pd" => [], "qd" => []) for name in keys(data["load"]))
# Increase load bids
# 10% of the load (used for increasing quantities)
for name in keys(data["load"])
    _load = data["load"][name]
    _bids = rand(Binomial(1, _prob_bids), _data_size)

    load_bids_data[name]["pd"] = _load["pd"] .+ _load["pd"] .* extra_load_mult .* _bids
    load_bids_data[name]["qd"] = _load["qd"] .+ _load["qd"] .* extra_load_mult .* _bids
end


function set_data(data, load_bids, gen_cost, cost_unweighted, iter)
   newdata = deepcopy(data)
   for k in keys(newdata["load"])
       newdata["load"][k]["pd"] = load_bids[k]["pd"][iter]
       newdata["load"][k]["qd"] = load_bids[k]["qd"][iter]
   end

   for k in keys(newdata["gen"])
       if data["gen"][k]["pmax"] > 0
           newdata["gen"][k]["cost"] = [gen_cost[k][1][iter], gen_cost[k][2][iter], 0]
           newdata["gen"][k]["cost_unweighted"] = [cost_unweighted[k][1][iter], cost_unweighted[k][2][iter], 0]
       end
   end
   return newdata
end


for iter in 1:1

    newdata = set_data(data, load_bids_data, gen_cost_data, unweighted_cost_data, iter)
    # demand_file = open("demands.txt", "r")

    # for lines in readlines(demand_file)
        # print the line
        # println(lines)

    # end

    print(keys(newdata["load"]))
    print("\n")
    print(length(keys(newdata["load"])))
    print("\n")
    for k in keys(newdata["load"])
        print(newdata["load"][k]["pd"])
        print("\n")
    end

    ### To set the active and reactive demand
    #
    # newdata["load"][k]["p(q)d"] = app
    #
    # , where k is the bus index
    #
    ###

    opf_sol = PowerModels.run_ac_opf(newdata, solver, setting = Dict("output" => Dict("branch_flows" => true)))
    

    if opf_sol["termination_status"] == LOCALLY_SOLVED
        random_file_name = rand(1:1000000)
        fileout =  open(outdir * string(random_file_name) * ".json", "w")
        write(fileout, "{")
        # Retrieve: (p^d, q^d) and (p^g, v)
        res  = Dict{String, Any}()
        res["prev_day_T"] = prev_day_T[iter]
        res["next_day_T"] = next_day_T[iter]
        res["pd"] = Dict(name => load["pd"] for (name, load) in newdata["load"])
        print(res["pd"])
        res["qd"] = Dict(name => load["qd"] for (name, load) in newdata["load"])
        res["gcost"] = Dict(name => gen["cost"] for (name, gen) in newdata["gen"])
        res["vg"] = Dict(name => opf_sol["solution"]["bus"][string(gen["gen_bus"])]["vm"]
                                    for (name, gen) in newdata["gen"])
                                        #if data["gen"][name]["pmax"] > 0)
        res["pg"] = Dict(name => gen["pg"] for (name, gen) in opf_sol["solution"]["gen"])
                                        #if data["gen"][name]["pmax"] > 0)
        res["qg"] = Dict(name => gen["qg"] for (name, gen) in opf_sol["solution"]["gen"])
                                        #if data["gen"][name]["pmax"] > 0)

        # Lines
        res["pt"] = Dict(name => data["pt"] for (name, data) in opf_sol["solution"]["branch"])
        res["pf"] = Dict(name => data["pf"] for (name, data) in opf_sol["solution"]["branch"])
        res["qt"] = Dict(name => data["qt"] for (name, data) in opf_sol["solution"]["branch"])
        res["qf"] = Dict(name => data["qf"] for (name, data) in opf_sol["solution"]["branch"])

        # Buses
        res["va"] = Dict(name => data["va"] for (name, data) in opf_sol["solution"]["bus"])
        res["vm"] = Dict(name => data["vm"] for (name, data) in opf_sol["solution"]["bus"])
        res["objective"] = opf_sol["objective"]
        res["solve_time"] = opf_sol["solve_time"]
        push!(res_stack, res)

        # writing data (Va, Qg, Obj, Pg, Vm, Qd and Pd) on output file
        #values_va = collect(values(res["va"]))
        #string_va = JSON.json(res["va"])
        res["va"] = Dict(parse(Int, k) => v for (k, v) in res["va"])
        res["va"] = sort(collect(res["va"]), by = x->x[1]) # sort the dictionare by the key, in increasing order
        res["va"] = [pair[2] for pair in res["va"]]  # collect the values and remove the keys

        values_va = collect(values(res["va"]))
        string_va = JSON.json(values_va)
        string_va = replace(string_va, "{" => "[")
        string_va = replace(string_va, "}" => "]")
        write(fileout, "\"va\": " * string_va * ",")

        res["qg"] = Dict(parse(Int, k) => v for (k, v) in res["qg"])
        res["qg"] = sort(collect(res["qg"]), by = x->x[1])
        res["qg"] = [pair[2] for pair in res["qg"]]
        values_qg = collect(values(res["qg"]))
        string_qg = JSON.json(values_qg)
        string_qg = replace(string_qg, "{" => "[")
        string_qg = replace(string_qg, "}" => "]")
        write(fileout, "\"qg\": " * string_qg * ",")

        ##

        ######## EXTRACTION WITHOUT ORDERING

        ##

        #values_qg = collect(values(res["qg"]))
        #string_qg = JSON.json(values_qg)
        #string_qg = replace(string_qg, "{" => "[")
        #string_qg = replace(string_qg, "}" => "]")
        #write(fileout, "\"qg\": " * string_qg * ",")

        #res["objective"] = Dict(parse(Int, k) => v for (k, v) in res["objective"])
        #res["objective"] = sort(collect(res["objective"]), by = x->x[1])
        #res["objective"] = [pair[2] for pair in res["objective"]]
        values_obj = collect(values(res["objective"]))
        string_obj = JSON.json(values_obj)
        string_obj = replace(string_obj, "{" => "[")
        string_obj = replace(string_obj, "}" => "]")
        write(fileout, "\"obj\": " * string_obj * ",")


        #values_obj = collect(values(res["objective"]))
        #string_obj = JSON.json(values_obj)
        #string_obj = replace(string_obj, "{" => "")
        #string_obj = replace(string_obj, "}" => "")
        #write(fileout, "\"obj\": " * string_obj * ",")

        res["pg"] = Dict(parse(Int, k) => v for (k, v) in res["pg"])
        res["pg"] = sort(collect(res["pg"]), by = x->x[1])
        res["pg"] = [pair[2] for pair in res["pg"]]
        values_pg = collect(values(res["pg"]))
        string_pg = JSON.json(values_pg)
        string_pg = replace(string_pg, "{" => "[")
        string_pg = replace(string_pg, "}" => "]")
        write(fileout, "\"pg\": " * string_pg * ",")

        #values_pg = collect(values(res["pg"]))
        #string_pg = JSON.json(values_pg)
        #string_pg = replace(string_pg, "{" => "[")
        #string_pg = replace(string_pg, "}" => "]")
        #write(fileout, "\"pg\": " * string_pg * ",")

        res["vm"] = Dict(parse(Int, k) => v for (k, v) in res["vm"])
        res["vm"] = sort(collect(res["vm"]), by = x->x[1])
        res["vm"] = [pair[2] for pair in res["vm"]]
        values_vm = collect(values(res["vm"]))
        string_vm = JSON.json(values_vm)
        string_vm = replace(string_vm, "{" => "[")
        string_vm = replace(string_vm, "}" => "]")
        write(fileout, "\"vm\": " * string_vm * ",")

        res["qd"] = Dict(parse(Int, k) => v for (k, v) in res["qd"])
        res["qd"] = sort(collect(res["qd"]), by = x->x[1])
        res["qd"] = [pair[2] for pair in res["qd"]]
        values_qd = collect(values(res["qd"]))
        string_qd = JSON.json(values_qd)
        string_qd = replace(string_qd, "{" => "[")
        string_qd = replace(string_qd, "}" => "]")
        write(fileout, "\"qd\": " * string_qd * ",")

        res["pd"] = Dict(parse(Int, k) => v for (k, v) in res["pd"])
        res["pd"] = sort(collect(res["pd"]), by = x->x[1])
        res["pd"] = [pair[2] for pair in res["pd"]]
        values_pd = collect(values(res["pd"]))
        string_pd = JSON.json(values_pd)
        string_pd = replace(string_pd, "{" => "[")
        string_pd = replace(string_pd, "}" => "]")
        write(fileout, "\"pd\": " * string_pd * ",")


        values_prev_day_T = collect(values(res["prev_day_T"]))
        string_prev_day_T = JSON.json(values_prev_day_T)
        string_prev_day_T = replace(string_prev_day_T, "{" => "")
        string_prev_day_T = replace(string_prev_day_T, "}" => "")
        write(fileout, "\"feat\": [" * string_prev_day_T * " , ")


        values_next_day_T = collect(values(res["next_day_T"]))
        string_next_day_T = JSON.json(values_next_day_T)
        string_next_day_T = replace(string_next_day_T, "{" => "")
        string_next_day_T = replace(string_next_day_T, "}" => "")
        write(fileout, string_next_day_T * "]")

        write(fileout, "}")

        close(fileout)

    end

end
