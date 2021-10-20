import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
BigFreq = 99999
BigCap = 99999
LineCap = 30
TransDistance = 0.1
EPS = 1e-9

def generate_station_dict_from_line_dict(line_dict):
    station_dict = {station: [] for station in station_loc.keys()}
    for line_name, line in line_dict.items():
        for s in line:
            station_dict[s].append(line_name)
    return station_dict

def assign_freq(graph, station_dict, line_freq, line_dict):
    for station, lines in station_dict.items():
        for line_name in lines:
            graph[str(station)+'_enter'][str(station)+'_'+line_name]['frequency'] = line_freq[line_name]
    for line_name, line in line_dict.items():
        for station_idx in range(len(line) - 1):
            s1 = line[station_idx]
            s2 = line[station_idx + 1]
            graph[str(s1) + '_' + line_name][str(s2) + '_' + line_name]['frequency'] = line_freq[line_name]
    return graph


def genearte_freq_graph(line_dict, station_loc, distance_dict, line_freq):
    # input: line info dict, distance matrix/dict
    # output: a modified graph, one station is replaced by one entrance, one exit, several
    G = nx.DiGraph()
    station_dict = generate_station_dict_from_line_dict(line_dict)
    # step 1. generate in-vehicle travel links
    for line_name, line in line_dict.items():
        G.add_node(str(line[0])+'_'+line_name,loc=station_loc[line[0]])
        for station_idx in range(len(line) - 1):
            s1 = line[station_idx]
            s2 = line[station_idx + 1]
            G.add_node(str(s2)+'_'+line_name,loc = station_loc[s2])
            G.add_edge(str(s1)+'_'+line_name,str(s2)+'_'+line_name, line_name=line_name, length=distance_dict[(s1, s2)], frequency = BigFreq, capacity=LineCap,type='inv')
    # step 2. generate boarding/alighting links
    for station, st_loc in station_loc.items():
        G.add_node(str(station)+'_enter')
        G.add_node(str(station)+'_exit')
        for line_name in station_dict[station]:
            G.add_edge(str(station)+'_enter', str(station)+'_'+line_name, line_name=line_name, length=0, frequency = BigFreq, capacity=BigCap,type='boarding')
            G.add_edge(str(station)+'_'+line_name, str(station)+'_exit', line_name=line_name, length =0, frequency = BigFreq, capacity=BigCap,type='alighting')
            G.add_edge(str(station)+'_exit', str(station)+'_enter',line_name='black0',length = TransDistance,frequency = BigFreq, capacity=BigCap,type='transfer')
    # step 3. assign frequencies
    G = assign_freq(G, station_dict, line_freq, line_dict)
    return G


def solve_transit_assignment_LP(G,OD,problem_name='0'):
    model = gp.Model('transit_assign'+problem_name)
    set_OD = list(OD.keys())
    set_D = list(set([od[1] for od in set_OD]))
    set_Arc = list(G.edges)
    set_Node = list(G.nodes)
    v = model.addVars(set_D, set_Arc, name='v', lb=0, ub=BigCap)
    w = model.addVars(set_D, set_Node, name='w', lb=0)
    OBJ_W = gp.quicksum([w[d, i] for d in set_D for i in set_Node])
    #import pdb;pdb.set_trace()
    OBJ_V = gp.quicksum([v[d, a[0],a[1]]*G[a[0]][a[1]]['length'] for d in set_D for a in set_Arc if G[a[0]][a[1]]['type']=='inv'])
    model.setObjective(OBJ_V+OBJ_W, GRB.MINIMIZE)
    def g_demand_func(o,d):
        if (o,d) in OD.keys():
            #print(o,d,OD[(o,d)])
            return OD[(o,d)]
            #return 1
        else:
            return 0.

    for i in set_Node:
        for d in set_D:
            if i!=d:
                model.addConstr(
                    (gp.quicksum([v[d,i,j] for j in list(G.successors(i))]) - gp.quicksum([v[d,j,i] for j in list(G.predecessors(i))]) == g_demand_func(i,d)),
                    name='flow_conserv,'+i+','+d)

    for d in set_D:
        for i in set_Node:
            for j in list(G.successors(i)):
                if G[i][j]['type']=='boarding':
                    model.addConstr(
                        v[d,i,j]<=G[i][j]['frequency']*w[d, i],
                        name='wait_time,'+d+','+i+','+j
                    )
    for a in set_Arc:
        if G[a[0]][a[1]]['type']=='inv':
             model.addConstr(
                (gp.quicksum([v[d,a[0],a[1]] for d in set_D]) <= G[a[0]][a[1]]['frequency'] * G[a[0]][a[1]]['capacity']),
                name='capacity,'+a[0]+','+a[1]
             )
    model.optimize()
    return model,v,w


def assign_vd_2_vod(v_val, OD, G):
    set_OD = list(OD.keys())
    set_D = list(set([od[1] for od in set_OD]))
    v_od_dict = dict()
    # to assign vd to vod, we first identify the subgraph with vd>0, which should be acyclic DAG,
    # we do topological sorting and assign each OD from the O's topological order to the end.
    # Starting from o, we should assign to the flow to the neighbor node with the smallest topological order
    for d in set_D:
        selected_edges = [(u,v) for u,v in G.edges() if v_val[d,u,v]>EPS]
        sg_d = G.edge_subgraph(selected_edges)
        assert nx.algorithms.dag.is_directed_acyclic_graph(sg_d)
        topo_sort = list(nx.algorithms.dag.topological_sort(sg_d))
        for (o,d1) in set_OD:
            if d1==d:
                temp_sorted = topo_sort[topo_sort.index(o):]
                V_od_val = {node:0. for node in temp_sorted}
                V_od_val[o] = OD[(o,d)]
                v_od_val_edge = dict()
                for node in temp_sorted:
                    if V_od_val[node]>EPS:
                        next_nodes = list(sg_d.successors(node))
                        total_d_flow = sum([v_val[d,node,node1] for node1 in next_nodes])
                        for node2 in next_nodes:
                            this_flow = V_od_val[node] * v_val[d,node,node2]/total_d_flow
                            V_od_val[node2] += this_flow
                            if (node,node2) in v_od_val_edge.keys():
                                v_od_val_edge[(node,node2)]+= this_flow
                            else:
                                v_od_val_edge[(node,node2)] = this_flow
                v_od_dict[(o,d1)] = v_od_val_edge
                assert abs(V_od_val[d]-OD[(o,d)])<EPS
    return v_od_dict






    return


if __name__ =='__main__':
    # name of line and station: underscore is not allowed
    G0 = nx.MultiDiGraph()
    np.random.seed(1)
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
    station_loc = {i: (np.random.rand(), np.random.rand()) for i in range(1, 18)}
    distance_dict = {(s1, s2): ((station_loc[s1][0] - station_loc[s2][0]) ** 2 + (
                station_loc[s1][1] - station_loc[s2][1]) ** 2) ** 0.5 for s1 in range(1, 18) for s2 in range(1, 18)}
    line_freq_dict = {line_name: 10. for line_name in line_dict.keys()}

    for i, loc in station_loc.items():
        G0.add_node(i, loc=loc)
    for line_name, line in line_dict.items():
        for station_idx in range(len(line) - 1):
            s1 = line[station_idx]
            s2 = line[station_idx + 1]
            G0.add_edge(s1, s2, line_name=line_name, length=distance_dict[(s1, s2)])
    OD = (15 * np.random.rand(17, 17)).astype(int)
    OD = {(str(i + 1) + '_enter', str(j + 1) + '_exit'): OD[i, j] for i in range(OD.shape[0]) for j in
               range(OD.shape[1]) if (OD[i, j] > 0 and i != j)}

    # colors = [G[u][v][w]['line_name'][:-1] for u,v,w in G.edges]
    # pos = nx.spring_layout(G)
    # nx.draw(G,pos,with_labels=True, edge_color = colors, connectionstyle='arc3, rad = 0.1')

    G = genearte_freq_graph(line_dict, station_loc, distance_dict, line_freq_dict)
    # colors = [G[u][v]['line_name'][:-1] for u,v in G.edges]
    # pos = nx.spring_layout(G)
    # nx.draw(G,pos,with_labels=True, edge_color = colors, connectionstyle='arc3, rad = 0.1')
    model,v,w = solve_transit_assignment_LP(G, OD, problem_name='0')
    cons_nonzeroPI = {cons.ConstrName:cons.PI for cons in model.getConstrs() if abs(cons.PI)>1e-10 and cons.ConstrName.split(',')[0]=='capacity'}
    v_val = {key: val.X for key, val in v.items()}
    v_od_dict = assign_vd_2_vod(v_val, OD, G)


