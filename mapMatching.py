import geopandas as gpd
import os
import numpy as np
import pandas as pd
from shapely import Point
from shapely import LineString
from shapely import Polygon
from joblib import Parallel, delayed
import time

from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.simple import SimpleMatcher


street_graph = os.path.join("..","gis","streetGraphs")
point_traces = os.path.join("..","gis","pointTraces")
cleaned_travel = os.path.join("..","gis","cleanedTravelData")
matched_folder = os.path.join("..","gis","matchedTraces")

subfolder_name = ["philadelphia","beijing"]
utm_crs = ["EPSG:32618","EPSG:32650"]
crs_filename_append = [s.split(":")[1] for s in utm_crs]

def clip_network(trace_gdf,edges_gdf,nodes_gdf):
    buffer_radius = 300
    points = list(trace_gdf.geometry)
    if len(points)<=1:
        buffered = trace_gdf.buffer(buffer_radius)
    else:
        line = LineString(points)
        line_gdf = gpd.GeoSeries([line], crs=trace_gdf.crs)
        buffered = line_gdf.buffer(buffer_radius)
    clipped_edges_gdf = edges_gdf[edges_gdf.intersects(buffered.union_all())]
    #now get all nodes for this
    all_node_ids = pd.unique(clipped_edges_gdf[["u","v"]].values.ravel())
    clipped_nodes_gdf = nodes_gdf[nodes_gdf["nodeID"].isin(all_node_ids)]
    return clipped_edges_gdf,clipped_nodes_gdf

def map_match(trace_gdf,clipped_edges_gdf,clipped_nodes_gdf):
    #creating the leuven map matching map
    map_con = InMemMap("graph", use_latlon=False, use_rtree=True, index_edges=True)
    for nid, row in clipped_nodes_gdf.iterrows():
        map_con.add_node(row["nodeID"], (row.geometry.y, row.geometry.x))
    for nid, row in clipped_edges_gdf.iterrows():
        map_con.add_edge(row["u"], row["v"])
    #our path
    path = list(zip(trace_gdf.geometry.y,trace_gdf.geometry.x))

    #the parameters here are critcal for a good result. a lot of tinkering was done. non_emitting_states and only_edges == True is very important
    #having a bigger buffer in the clipping and also bigger max_dist is the most important. makes it slower but we have time
    matcher = DistanceMatcher(map_con, max_dist=300, obs_noise=5, obs_noise_ne=10, non_emitting_states=True,only_edges=True, max_lattice_width=20)
    states, _ = matcher.match(path)
    nodes = matcher.path_pred_onlynodes
    return nodes

def nodes_to_gdf(nodes,edge_gdf):
    edge_list = list(zip(nodes[:-1],nodes[1:]))
    edge_list_df = pd.DataFrame(edge_list, columns=["u","v"])
    edge_list_df["order"] = range(len(edge_list_df))
    matched_edges = edge_list_df.merge(edge_gdf,on=["u","v"],how="left")
    matched_edges = matched_edges.sort_values("order").drop(columns="order")
    matched_edges_gdf = gpd.GeoDataFrame(matched_edges, geometry="geometry", crs=edge_gdf.crs)
    return matched_edges_gdf


def for_loop_function(file_path,trace_gdf,edges_gdf,nodes_gdf):
    print(file_path)
    clipped_edges_gdf,clipped_nodes_gdf = clip_network(trace_gdf,edges_gdf,nodes_gdf)
    nodes = map_match(trace_gdf,clipped_edges_gdf,clipped_nodes_gdf)
    route_gdf = nodes_to_gdf(nodes,clipped_edges_gdf)
    return (file_path,route_gdf)

def iterate_and_match(file_list,location_i):
    path_edges = os.path.join(street_graph,subfolder_name[location_i],"edges_"+crs_filename_append[location_i]+".shp")
    path_nodes = os.path.join(street_graph,subfolder_name[location_i],"nodes_"+crs_filename_append[location_i]+".shp")

    edges_gdf = gpd.read_file(path_edges)
    nodes_gdf = gpd.read_file(path_nodes)

    target_folder = os.path.join(matched_folder,subfolder_name[location_i])
    os.makedirs(target_folder, exist_ok=True)

    cut_off = 100
    #cut_off = len(file_list)

    start = time.time()

    store_in_memory = {file_path:gpd.read_file(file_path) for file_path in file_list[:cut_off]}

    #we use parallelization here because it would take forever otherwise
    return_tuples = Parallel(n_jobs=10)(delayed(for_loop_function)(x,store_in_memory[x],edges_gdf,nodes_gdf) for x in file_list[:cut_off])
    
    for x in return_tuples:
        file_path,route_gdf = x
        filename = os.path.basename(file_path)
        route_gdf.to_file(os.path.join(target_folder,filename) )


    end = time.time()
    print(f"Execution time: {end - start:.4f} seconds")

def add_times(location_i):
    target_folder = os.path.join(matched_folder,subfolder_name[location_i])
    files = [f for f in os.listdir(target_folder) if f.endswith(".shp")]
    #add times from tripsGDF
    if location_i == 0:
        trip_path = os.path.join(cleaned_travel,subfolder_name[location_i],"tripsCleaned_32618.shp")
        trip_gdf = gpd.read_file(trip_path)
        trip_gdf["start_date_time"] = pd.to_datetime(trip_gdf['Start'],format='ISO8601')
        trip_gdf["end_date_time"] = pd.to_datetime(trip_gdf['Stop'],format='ISO8601')
        for file in files:
            tripID = int(file.split("_")[1].split(".")[0])
            print(tripID)
            file = os.path.join(target_folder,file)
            route_gdf = gpd.read_file(file)
            start = trip_gdf.loc[trip_gdf["TripID"] == tripID]["start_date_time"].min()
            end =  trip_gdf.loc[trip_gdf["TripID"] == tripID]["end_date_time"].max()
            even_spaced = pd.date_range(start,end,periods=len(route_gdf))
            route_gdf["date_time"] = even_spaced
            route_gdf["date"] = route_gdf["date_time"].dt.date
            route_gdf["time"] = route_gdf["date_time"].dt.time
            route_gdf = route_gdf.drop(["date_time"],axis=1)
            route_gdf.to_file(file)

    #for beijing, times directly from the traces, also added to gdf 
    elif location_i == 1:
        trip_path = os.path.join(cleaned_travel,subfolder_name[location_i],"trips.csv")
        trip_df = pd.read_csv(trip_path)
        trip_df['Start'] = pd.NaT
        trip_df['End'] = pd.NaT
        
        for file in files:
            file = os.path.join(target_folder,file)
            route_gdf = gpd.read_file(file)
            trace_path = os.path.join(point_traces,subfolder_name[location_i],os.path.basename(file))
            trace_gdf = gpd.read_file(trace_path)
            trace_gdf["date_time"] = trace_gdf["date"] + " " +trace_gdf["time"]
            trace_gdf["date_time"] = pd.to_datetime(trace_gdf['date_time'],format='ISO8601')
            start = trace_gdf["date_time"].min()
            end = trace_gdf["date_time"].max()
            even_spaced = pd.date_range(start,end,periods=len(route_gdf))
            route_gdf["date_time"] = even_spaced
            route_gdf["date"] = route_gdf["date_time"].dt.date
            route_gdf["time"] = route_gdf["date_time"].dt.time
            route_gdf = route_gdf.drop(["date_time"],axis=1)
            route_gdf.to_file(file)
            tripID = int((os.path.basename(file).split("_")[1]).split(".")[0])
            match_idx = trip_df[trip_df['TripID'] == tripID].index
            if not match_idx.empty:
                trip_df.at[match_idx[0], 'Start'] = start
                trip_df.at[match_idx[0], 'End'] = end
        trip_df.to_csv(trip_path,index=False)

def get_philly_files(location_i):
    path = os.path.join(point_traces,subfolder_name[location_i])
    files = [f for f in os.listdir(path) if f.endswith(".shp")]
    return [os.path.join(path, f) for f in files]

#old
#def get_beijing_files(location_i):
#    our_label = "car"
#    path = os.path.join(cleaned_travel,subfolder_name[location_i],"trips.csv")
#    label_df = pd.read_csv(path)
#    trip_ids = list(label_df[label_df["label"] == our_label]["tripID"])
#    return [os.path.join(point_traces, subfolder_name[location_i],   "tra_"+str(trip_id)+".shp"    ) for trip_id in trip_ids]


get_file_paths = [get_philly_files,get_philly_files]

location_i = 1
def main():
    print(subfolder_name[location_i])
    file_list = get_file_paths[location_i](location_i) 
    iterate_and_match(file_list,location_i)
    add_times(location_i)

if __name__ == "__main__":
    main()

