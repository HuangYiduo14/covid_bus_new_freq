import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from global_var_define import *

def solve_transit_assignment_LP(G,OD,problem_name='0'):
    # G is a nx.BiGraph, OD is a dictionary
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
                        v[d,i,j]-G[i][j]['frequency']*w[d, i]<=0.,
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
    # the assignment problem is solved with respect to v^d vars, now we want to get OD flows
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
                V_od_node = {node:0. for node in temp_sorted} #V_od_node is the throughput flow of one node
                V_od_node[o] = OD[(o,d)]
                v_od_edge = dict() # v_od_edge is the flow on certain edge
                for node in temp_sorted:
                    if V_od_node[node]>EPS:
                        next_nodes = list(sg_d.successors(node))
                        total_d_flow = sum([v_val[d,node,node1] for node1 in next_nodes])
                        for node2 in next_nodes:
                            this_flow = V_od_node[node] * v_val[d,node,node2]/total_d_flow
                            V_od_node[node2] += this_flow
                            if (node,node2) in v_od_edge.keys():
                                v_od_edge[(node,node2)]+= this_flow
                            else:
                                v_od_edge[(node,node2)] = this_flow
                v_od_dict[(o,d1)] = v_od_edge
                assert abs(V_od_node[d]-OD[(o,d)])<EPS
    return v_od_dict





