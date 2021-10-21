import networkx as nx
from global_var_define import *


def generate_physical_graph(station_loc, line_dict, distance_dict):
    # generate physical graph: an example: for drawing only
    G0 = nx.MultiDiGraph()
    for i, loc in station_loc.items():
        G0.add_node(i, loc=loc)
    for line_name, line in line_dict.items():
        for station_idx in range(len(line) - 1):
            s1 = line[station_idx]
            s2 = line[station_idx + 1]
            G0.add_edge(s1, s2, line_name=line_name, length=distance_dict[(s1, s2)])
    # colors = [G0[u][v][w]['line_name'][:-1] for u,v,w in G.edges]
    # pos = nx.spring_layout(G)
    # nx.draw(G0,pos,with_labels=True, edge_color = colors, connectionstyle='arc3, rad = 0.1')
    return G0


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
def generate_station_dict_from_line_dict(line_dict, station_list):
    station_dict = {station: [] for station in station_list}
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
    station_dict = generate_station_dict_from_line_dict(line_dict,station_loc.keys())
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




