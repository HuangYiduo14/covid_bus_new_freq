import numpy as np
import networkx as nx
from approx_transit_assignment import solve_transit_assignment_LP, assign_vd_2_vod
from generate_graph import genearte_freq_graph, generate_physical_graph
import pandas as pd
from global_var_define import *

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
    # solve the assignment problem and assign OD
    model,v,w = solve_transit_assignment_LP(G, OD, problem_name='0')
    #cons_nonzeroPI = {cons.ConstrName:cons.PI for cons in model.getConstrs() if abs(cons.PI)>1e-10 and cons.ConstrName.split(',')[0]=='capacity'}
    v_val = {key: val.X for key, val in v.items()}
    v_od_dict = assign_vd_2_vod(v_val, OD, G)



