import numpy as np
from generate_graph import genearte_freq_graph, generate_physical_graph, generate_station_dict_from_line_dict
from frequency_design import get_result
import pandas as pd
from global_var_define import *
import multiprocessing
import tqdm
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


    def obj_func(line_freq_dict1):
        return get_result(G, line_freq_dict1, station_dict, line_dict, OD, SIR_location_table)



    cand_set = [10.,11.,12.,13.,14.,15.,16.,20.]
    cand_list = []
    cand_num_list = []
    for f_red in cand_set:
        for f_blue in cand_set:
            for f_yellow in cand_set:
                for f_green in cand_set:
                    cand_list.append(
                        {
                            'red0':f_red,'blue0':f_blue,'green0':f_green,'yellow0':f_yellow,
                            'red1': f_red, 'blue1': f_blue, 'green1': f_green, 'yellow1': f_yellow
                         }
                    )
                    cand_num_list.append([f_red,f_blue,f_green,f_yellow])
    all_results = []
    #with multiprocessing.Pool(cpu_count) as pool:
    #    for result in tqdm.tqdm(pool.imap_unordered(obj_func, cand_list),total=len(cand_list)):
    for cand_freq in cand_list:
        print(cand_freq,'--'*20)
        result = obj_func(cand_freq)
        print(result)
        all_results.append(result)

    result_df = pd.DataFrame([cand_num_list[i]+[all_results[i]] for i in range(len(cand_list))],columns=['red','blue','green','yellow','encounters'])
    result_df.to_csv('exp_results0.csv')




