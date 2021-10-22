import numpy as np
import networkx as nx
import pandas as pd
from global_var_define import *
from approx_transit_assignment import solve_transit_assignment_LP, assign_vd_2_vod
from generate_graph import assign_freq
from gurobipy import GRB



def get_result(G, line_freq_dict, station_dict, line_dict, OD, SIR_location_table):
    G = assign_freq(G, station_dict, line_freq_dict, line_dict)
    model, v, w = solve_transit_assignment_LP(G, OD, problem_name='0')
    #import pdb;pdb.set_trace()
    if model.getAttr(GRB.Attr.Status)!=2:
        v_val = {key: 0. for key, val in v.items()}
        #import pdb;pdb.set_trace()
        hhh = [[e[0],e[1]] for e in G.edges]
        v_od_df = pd.DataFrame([[od[0], od[1], hh[0],hh[1],0.] for od in OD.keys() for hh in hhh[:3]],columns=['o','d','i','j','flow'])
    else:
        v_val = {key: val.X for key, val in v.items()}
        v_od_dict = assign_vd_2_vod(v_val, OD, G)
        v_od_dict_full = [[i[0], i[1], j[0], j[1], v_od_dict[i][j]] for i in v_od_dict.keys() for j in v_od_dict[i].keys()]
        v_od_df = pd.DataFrame(v_od_dict_full, columns=['o', 'd', 'i', 'j', 'flow'])
    total_encounter, _, _, _, _ = calculate_covid_contact(G, v_od_df, SIR_location_table)
    return total_encounter



# inline flow, outline flow and exit flow: consider one station and a line

#       --inline flow --> (station_line) -- outline flow-->
# (station_enter) -- boarding flow --/    \-- exit flow --> (station_exit)

def calculate_covid_contact(G: nx.DiGraph, v_od_df: pd.DataFrame, SIR_location_table: pd.DataFrame):
    # v_od_df = pd.DataFrame(v_od_dict_full, columns=['o','d','i','j','flow'])
    # SIR_location_table = pd.DataFrame(columns=['S','I','R','N','S_rate','I_rate']) with node as index
    contact_number = 0

    # step 0. calculate v
    v_all_df = nx.convert_matrix.to_pandas_edgelist(G)
    v_od_df['I_flow'] = v_od_df['flow'].values * (SIR_location_table.loc[v_od_df['o'].str.split('_').str[0].astype(int),'I_rate']).values
    v_od_df['S_flow'] = v_od_df['flow'].values * (SIR_location_table.loc[v_od_df['o'].str.split('_').str[0].astype(int),'S_rate']).values
    v_od_df_grouped = v_od_df[['i','j','flow','I_flow','S_flow']].groupby(['i','j']).sum()
    v_all_df = pd.merge(v_all_df, v_od_df_grouped, left_on=['source','target'],right_index=True,how='left')
    v_all_df.fillna(0,inplace=True)

    # step 1. calculate on-board infections
    v_invehicle_df = v_all_df.loc[v_all_df['type']=='inv']
    count_encouter_invehicle = v_invehicle_df['I_flow']*v_invehicle_df['S_flow']*v_invehicle_df['length']
    contact_invehicle = count_encouter_invehicle.sum()

    # step 2. calculate platform waiting time and infections
    # aggregate v_I flows for each platforms
    v_boarding_platform = v_all_df.loc[v_all_df['type']=='boarding'].copy()
    v_boarding_platform['station'] = v_boarding_platform['source'].str.split('_').str[0]
    v_boarding_platform = pd.merge(v_boarding_platform,
                                   v_boarding_platform[['station','I_flow']].groupby('station').sum(),left_on='station',right_index=True,how='left',suffixes=('','_platform'))
    # we need to calculate different flows near the platform to estimate the probability of failing-to-board
    v_boarding_platform['inline_flow'] = 0.
    v_boarding_platform['outline_flow'] = 0.
    v_boarding_platform['exit_flow'] = 0.

    for i in v_boarding_platform.index:
        this_node = v_boarding_platform.loc[i,'target']
        pre_nodes = list(G.predecessors(this_node))
        succ_nodes = list(G.successors(this_node))

        inline_nodes = [node for node in pre_nodes if node.split('_')[1]!='enter']
        outline_nodes = [node for node in succ_nodes if node.split('_')[1]!='exit']
        exit_nodes = [node for node in succ_nodes if node.split('_')[1]=='exit']

        if len(inline_nodes)>0:
            inline_flow = v_all_df.loc[(v_all_df['source']==inline_nodes[0])&(v_all_df['target']==this_node),'flow'].sum()
        else:
            inline_flow = 0.
        if len(outline_nodes)>0:
            outline_flow = v_all_df.loc[
                (v_all_df['source'] == this_node) & (v_all_df['target'] == outline_nodes[0]), 'flow'].sum()
        else:
            outline_flow = 0.
        exit_flow =  v_all_df.loc[
                (v_all_df['source'] == this_node) & (v_all_df['target'] == exit_nodes[0]), 'flow'].sum()
        v_boarding_platform.loc[i,'inline_flow'] = inline_flow
        v_boarding_platform.loc[i,'outline_flow'] = outline_flow
        v_boarding_platform.loc[i,'exit_flow'] = exit_flow
    # type 1. truncated geometrical distribution
    if DELAY_FUNC_TYPE == 'GEO':
        v_boarding_platform['P_success_boarding'] = (v_boarding_platform['capacity']*v_boarding_platform['frequency']-v_boarding_platform['inline_flow']+v_boarding_platform['outline_flow'])/(EPS+v_boarding_platform['flow'])
        v_boarding_platform.loc[v_boarding_platform['P_success_boarding'] > 1.,'P_success_boarding'] = 1.
        v_boarding_platform.loc[v_boarding_platform['P_success_boarding'] < EPS_BOARDING, 'P_success_boarding'] = EPS_BOARDING
        v_boarding_platform['P_fail_boarding'] = 1. - v_boarding_platform['P_success_boarding']
        v_boarding_platform['delay'] = 1. / v_boarding_platform['frequency']*(1./v_boarding_platform['P_success_boarding'])
    elif DELAY_FUNC_TYPE == 'BPR':
        # type 2. BPR formula
        v_boarding_platform['relative_flow'] = v_boarding_platform['flow']/(v_boarding_platform['capacity']*v_boarding_platform['frequency']-v_boarding_platform['inline_flow']+v_boarding_platform['outline_flow']+EPS)
        v_boarding_platform['delay'] = 1. / v_boarding_platform['frequency'] * (1 + BPR_ALPHA * (v_boarding_platform['relative_flow'])**BPR_BETA)
    count_encouter_platform = v_boarding_platform['delay']*v_boarding_platform['S_flow']*v_boarding_platform['I_flow_platform']
    contact_platform = count_encouter_platform.sum()
    return contact_platform+contact_invehicle, contact_platform, contact_invehicle, count_encouter_platform, count_encouter_invehicle
