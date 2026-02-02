import geopandas as gpd
import momepy
import networkx as nx
import numpy as np


def create_route_graph(edges_gdf):
    G_prime = momepy.gdf_to_nx(edges_gdf,approach="primal",multigraph=False,directed=True)
    G_dual = nx.line_graph(G_prime)
    #add the edgeID pair to each edge in the dual graph, since each dual edge now reprensents a trans behavior
    for edge in G_dual.edges():
        original_from_edge,original_to_edge = edge
        from_streetID = G_prime.get_edge_data(*original_from_edge)["edgeID"]
        to_streetID = G_prime.get_edge_data(*original_to_edge)["edgeID"]
        G_dual[original_from_edge][original_to_edge]["transition"] = (from_streetID,to_streetID)

    #for u, v, data in G_dual.edges(data=True):
    #    print(f"Edge from {u} to {v}: {data}")



    #add original u,v to nodes (which were edges)
    for node in G_dual.nodes(data=True):
        org_edge = node[0]
        org_u = G_prime.get_edge_data(*org_edge)["u"]
        org_v = G_prime.get_edge_data(*org_edge)["v"]
        G_dual.nodes[org_edge]["u"] = org_u
        G_dual.nodes[org_edge]["v"] = org_v
        G_dual.nodes[org_edge]["edgeID"] = edges_gdf[(edges_gdf["u"] == org_u) & (edges_gdf["v"] == org_v) ]["edgeID"].iloc[0]

    
    #for node, data in G_dual.nodes(data=True):
    #    print(f"Node {node}: {data}")

    return G_prime,G_dual

def add_weight_to_dual(G_dual,user_id,interval,weight_matrix,user_index_map,behav_index_map):
    user_index = user_index_map[user_id]
    weights = weight_matrix[user_index,:]

    for u,v,data in G_dual.edges(data=True):
        if interval == -1:
            trans = data.get("transition")
        else:
            trans = (interval,data.get('transition'))
        #try:
        weight = weights[behav_index_map[trans]]
        #except KeyError as e:
        #    print(u,v,data)
        #    print("This only happens in Beijing cause there are some overlapping nodes.")
        #    weight = 0
        data['weight'] = weight
    return G_dual

def add_special_nodes(G_dual,start_node_id,end_node_id,userID,interval,weight_matrix,user_index_map,behav_index_map):
    start_dual_target_nodes =  [(n,data) for n, data in G_dual.nodes(data=True) if data.get('u') == start_node_id]
    end_dual_target_nodes =  [(n,data) for n, data in G_dual.nodes(data=True) if data.get('v') == end_node_id]
    weights = weight_matrix[user_index_map[userID],:]

    G_dual.add_node("special_start_node")
    G_dual.add_node("special_end_node")

    for node in start_dual_target_nodes:
        if interval == -1:
            behav = node[1]["edgeID"]
        else:
            behav = (interval,node[1]["edgeID"])
        weight = weights[behav_index_map[behav]]
        G_dual.add_edge('special_start_node',node[0], weight=weight)

    for node in end_dual_target_nodes:
        if interval == -1:
            behav = node[1]["edgeID"]
        else:
            behav = (interval,node[1]["edgeID"])

        weight = weights[behav_index_map[behav]]
        G_dual.add_edge(node[0],'special_end_node', weight=weight)



def route(G_dual,G_prime,start_node_id,end_node_id,userID,interval,weight_matrix,user_index_map,behav_index_map,edges_gdf):
    add_special_nodes(G_dual,start_node_id,end_node_id,userID,interval,weight_matrix,user_index_map,behav_index_map)
    path = nx.dijkstra_path(G_dual, source='special_start_node', target='special_end_node', weight='weight')
    #remove special start and end
    path = path[1:-1]
    G_dual.remove_node('special_start_node')
    G_dual.remove_node('special_end_node')
    streetIDs =[ G_prime.get_edge_data(*edge)["edgeID"] for edge in path]
    #print( [ (u,v,behavs_index_map[(u,v)]) for u,v in zip(streetIDs[:-1],streetIDs[1:])    ]   )
    weights = [G_dual.get_edge_data(u,v)["weight"] for u,v in zip(path[:-1], path[1:])]
    #print(weights)
    gdf_result = edges_gdf[edges_gdf['edgeID'].isin(streetIDs)]
    gdf_result = gdf_result.set_index('edgeID').reindex(streetIDs).reset_index()
    return gdf_result


def shortest_path(G_prime,start_edge_id,end_edge_id,edges_gdf):
    edge_1 = None
    edge_2 = None
    for u, v, data in G_prime.edges(data=True):
        if data.get('edgeID') == start_edge_id:
            edge_1 = (u, v)
        if data.get('edgeID') == end_edge_id:
            edge_2 = (u, v)
    #can rarely happen because some edges disappear when road graph is created, edges that start and end at the same node, only one is kept. 
    #happens when a road segment represents a parking space and joins back to the real road all within one segment, f.e edgeID 71429
    if edge_1 == None or edge_2 == None:
        return None
    start_node = edge_1[0]  
    end_node = edge_2[1]    

    path = nx.dijkstra_path(G_prime, source=start_node, target=end_node, weight='length')

    edgeIDs = []
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        edge_data = G_prime.get_edge_data(u, v)
        # edge_data can be None if edge doesn't exist, but here it should exist
        edge_id = edge_data['edgeID']
        edgeIDs.append( edge_id)


    gdf_result = edges_gdf[edges_gdf['edgeID'].isin(edgeIDs)]
    gdf_result = gdf_result.set_index('edgeID').reindex(edgeIDs).reset_index()
    return gdf_result

def shortest_path_nodes(G_prime,start_node,end_node,edges_gdf):
    source_node = (start_node.geometry.x,start_node.geometry.y)
    target_node = (end_node.geometry.x,end_node.geometry.y)
    path = nx.dijkstra_path(G_prime, source=source_node, target=target_node, weight='length')
    
    edgeIDs = []
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        edge_data = G_prime.get_edge_data(u, v)
        # edge_data can be None if edge doesn't exist, but here it should exist
        edge_id = edge_data['edgeID']
        edgeIDs.append( edge_id)
        

    gdf_result = edges_gdf[edges_gdf['edgeID'].isin(edgeIDs)]
    gdf_result = gdf_result.set_index('edgeID').reindex(edgeIDs).reset_index()
    return gdf_result



