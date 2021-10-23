import numpy as np
from generate_graph import genearte_freq_graph, generate_physical_graph, generate_station_dict_from_line_dict
from frequency_design import get_result
import pandas as pd
from global_var_define import *
import multiprocessing
from scipy.sparse import hstack, csc_matrix, linalg, csr_matrix
import time
cpu_count = multiprocessing.cpu_count()

# node naming rules:
# [station name]_[line name]|[type name]
# example1: station1_red0 for the line node for line red for 0 direction at station 1
# example2: station1_enter is the entrance node at station 1

# arc type:
# line node -> line node: inv (in-vehicle transit)
# line node -> exit node: alighting
# enter node -> line node: boarding
# exit node -> enter node: transfer

# o node must be an enter node
# d node must be an exit node

if __name__ =='__main__':
    s_time = time.time()
    # name of line and station: underscore is not allowed
    # generate city map
    np.random.seed(RANDOM_SEED)
    line_dict = {
        'red0': [1, 2, 3, 4, 5, 6],
        'red1': [6, 5, 4, 3, 2, 1],
        'blue0': [7, 8, 9, 10, 11, 12],
        'blue1': [12, 11, 10, 9, 8, 7],
        'green0': [13, 14, 3, 9, 15, 16],
        'green1': [16, 15, 9, 3, 14, 13],
        'yellow0': [13, 14, 4, 10, 9, 15, 17],
        'yellow1': [17, 15, 9, 10, 4, 14, 13]
    }
    # generate sir data
    SIR_location_table = pd.DataFrame({
        'nodes':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
        'S':0.01*np.random.rand(17),
        'I':0.98*np.random.rand(17),
        'R':0.01*np.random.rand(17)
    })
    SIR_location_table = SIR_location_table.set_index('nodes')
    SIR_location_table['N'] = SIR_location_table['S'] + SIR_location_table['I'] + SIR_location_table['R']
    SIR_location_table['S_rate'] = SIR_location_table['S'] / SIR_location_table['N']
    SIR_location_table['I_rate'] = SIR_location_table['I'] / SIR_location_table['N']
    # generate station location, distance and initial frequency
    station_loc = {i: (np.random.rand(), np.random.rand()) for i in range(1, 18)}
    distance_dict = {(s1, s2): ((station_loc[s1][0] - station_loc[s2][0]) ** 2 + (
                station_loc[s1][1] - station_loc[s2][1]) ** 2) ** 0.5 for s1 in range(1, 18) for s2 in range(1, 18)}
    line_freq_dict = {line_name: 10. for line_name in line_dict.keys()}





    # generate OD demand as a dict
    OD = (15 * np.random.rand(17, 17)).astype(int)
    OD = {(str(i + 1) + '_enter', str(j + 1) + '_exit'): OD[i, j] for i in range(OD.shape[0]) for j in
               range(OD.shape[1]) if (OD[i, j] > 0 and i != j)}
    # use the info above to generate the physical and frequency graph
    G0 = generate_physical_graph(station_loc, line_dict, distance_dict)
    G = genearte_freq_graph(line_dict, station_loc, distance_dict, line_freq_dict)
    # calculate obj function
    station_dict = generate_station_dict_from_line_dict(line_dict, station_loc.keys())
    total_encounter,model,v,w = get_result(G, line_freq_dict, station_dict, line_dict, OD, SIR_location_table)


    var_basis = [var.index for var in model.getVars() if var.VBasis==0]

    constr_basis = [constr.index for constr in model.getConstrs() if constr.CBasis == 0]


    #print(len(var_basis),len(constr_basis))
    A = model.getA()
    A = csc_matrix(A)
    A_B = A[:,var_basis]
    col_ind_list = []
    row_ind_list = []
    for col_ind, row_ind in enumerate(constr_basis):
        col_ind_list.append(col_ind)
        row_ind_list.append(row_ind)
    N_slack = len(row_ind_list)
    data = [1. for _ in range(N_slack)]
    slack_coeff = csc_matrix( (data,(row_ind_list,col_ind_list)) ,shape=(A_B.shape[0],N_slack))
    A_B_full = hstack([A_B,slack_coeff])
    A_B_full = csc_matrix(A_B_full)
    #rhs = np.array(model.RHS) # verify this sparse matrix by comparing basis variable solutions
    #sol = linalg.spsolve(A_B_full,rhs)
    sol_base = [var.X for var in model.getVars() if var.VBasis==0]
    sol_base += [constr.Slack for constr in model.getConstrs() if constr.CBasis == 0]
    sol_base = np.array(sol_base)

    def construct_line_specific_rows(first_letter):
        line_specific_rows = {line_name: [] for line_name in line_freq_dict.keys()}
        for line_name in line_freq_dict.keys():
            for constr in model.getConstrs():
                name_constr = constr.ConstrName
                if name_constr[0] == first_letter:
                    j_node = constr.ConstrName.split(',')[-1]
                    j_line = j_node.split('_')[1]
                    if j_line == line_name:
                        line_specific_rows[line_name].append(constr.index)
        return line_specific_rows
    line_specific_boarding_rows = construct_line_specific_rows('w') # construct a dict {'line1':[row1,row2,...],..} each element is a list including row number of boarding time constraints for line l
    line_specific_capacity_rows = construct_line_specific_rows('c') # construct a dict {'line1':[row1,...],...}each element is a list including row number of capacity constraints for line l
    var_basis_ws = [var.index for var in model.getVars() if var.VBasis == 0 and var.VarName[0] == 'w']
    grad_all = dict()
    for line_name in line_freq_dict.keys():
        print(line_name)
        A_B_ws = A[:, var_basis_ws].copy()
        A_B_ws1 = csr_matrix((A_B_ws.shape[0],A_B_ws.shape[1]))
        A_B_ws[A_B_ws.nonzero()] = -1.
        A_B_ws = csr_matrix(A_B_ws)
        A_B_ws = A_B_ws[line_specific_boarding_rows[line_name],:]
        A_B_ws1[line_specific_boarding_rows[line_name],:] = A_B_ws # warning Performance
        A_B_ws1 = csc_matrix(
            hstack([
                csc_matrix((A_B.shape[0],A_B_full.shape[1]-N_slack - A_B_ws1.shape[1])),
                A_B_ws1,
                csc_matrix((A_B.shape[0],N_slack))
            ])
        )
        grad_1 = linalg.spsolve(A_B_full, A_B_ws1)
        grad_part_1 = - (grad_1.dot(sol_base))
        constr_grad = np.zeros(A_B_full.shape[0])
        constr_grad[line_specific_capacity_rows[line_name]] = LineCap
        grad_part_2 = linalg.spsolve(A_B_full, constr_grad) #constr_grad[l] is the flow gradient of rhs for line l
        grad = grad_part_1 + grad_part_2
        grad_all[line_name] = grad
    print('total time',time.time()-s_time)

    # def obj_func(line_freq_dict1):
    #     return get_result(G, line_freq_dict1, station_dict, line_dict, OD, SIR_location_table)
    #
    #
    #
    # cand_set = [10.,11.]
    # cand_list = []
    # cand_num_list = []
    # for f_red in cand_set:
    #     for f_blue in cand_set:
    #         for f_yellow in cand_set:
    #             for f_green in cand_set:
    #                 cand_list.append(
    #                     {
    #                         'red0':f_red,'blue0':f_blue,'green0':f_green,'yellow0':f_yellow,
    #                         'red1': f_red, 'blue1': f_blue, 'green1': f_green, 'yellow1': f_yellow
    #                      }
    #                 )
    #                 cand_num_list.append([f_red,f_blue,f_green,f_yellow])
    # all_results = []
    # #with multiprocessing.Pool(cpu_count) as pool:
    # #    for result in tqdm.tqdm(pool.imap_unordered(obj_func, cand_list),total=len(cand_list)):
    # for cand_freq in cand_list:
    #     print(cand_freq,'--'*20)
    #     result = obj_func(cand_freq)
    #     print(result)
    #     all_results.append(result)
    #
    # result_df = pd.DataFrame([cand_num_list[i]+[all_results[i]] for i in range(len(cand_list))],columns=['red','blue','green','yellow','encounters'])
    # result_df.to_csv('exp_results0.csv')




