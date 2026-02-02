import numpy as np
from collections import defaultdict
import pandas as pd
import geopandas as gpd
from itertools import compress
from shapely.ops import linemerge



def split_zero_non_zero_behavs(freq_data):
    adj_matrix = freq_data[0]
    behav_index_map = freq_data[1]
    #each all zero column (i.e. one behavior) will get False
    non_zero_mask = np.any(adj_matrix != 0,axis=0)
    #keys = np.array(list(behav_index_map.keys()))
    #values = np.array(list(behav_index_map.values()))

    #zero_index_map = dict(zip(keys[~non_zero_mask],values[~non_zero_mask]) )
    #non_zero_index_map = dict(zip(keys[non_zero_mask],values[non_zero_mask]))

    keys = list(behav_index_map.keys())
    values = list(behav_index_map.values())

    zero_index_map = dict(zip( compress(keys,~non_zero_mask),compress(values,~non_zero_mask)  ))
    non_zero_index_map = dict(zip( compress(keys,non_zero_mask),compress(values,non_zero_mask)  ))

    return zero_index_map,non_zero_index_map

def get_interval_i_behavs(freq_data,i,intervals):
    adj_matrix = freq_data[0]
    behav_index_map = freq_data[1]
    return (adj_matrix[:,i::intervals],dict(list(behav_index_map.items())[i::intervals]))

def split_app_trans(geo_data,freq_data,time_settings):
    is_time = time_settings[0]
    if not is_time:
        interval = 1
    else:
        interval = time_settings[1]
    edges_gdf = geo_data[1]
    
    split_point = len(edges_gdf)*interval
    print("split_point",split_point)
    adj_matrix = freq_data[0]
    behav_index_map = freq_data[1]
    app_matrix = adj_matrix[:,:split_point]
    trans_matrix = adj_matrix[:,split_point:]
    items = list(behav_index_map.items())
    print("split point-1",items[split_point-1])
    print("split point",items[split_point])
    app_dict = dict(items[:split_point])
    trans_dict = dict(items[split_point:])
    return (app_matrix,app_dict),(trans_matrix,trans_dict)

def app_behavs_to_geo(edges_gdf,index_map,time_settings):
    is_time = time_settings[0]
    if is_time:
        edge_list = [t[1] for t in list(index_map.keys())]
    else:
        edge_list = list(index_map.keys())
    return edges_gdf[edges_gdf["edgeID"].isin(edge_list)].reset_index()
   
def get_centroids(edges_gdf):
    centroids = edges_gdf.geometry.centroid
    return gpd.GeoDataFrame(edges_gdf.drop(columns='geometry'), geometry=centroids, crs=edges_gdf.crs)

def get_all_attribute_combinations(points_gdf,attributes):
    return points_gdf[attributes].drop_duplicates().values.tolist()

def get_zero_non_zero_att_combs(non_zero_points,zero_points,attributes):
    return get_all_attribute_combinations(non_zero_points,attributes),get_all_attribute_combinations(zero_points,attributes)

def filter_by_comb(points_gdf,comb,attributes):
    individual_masks = [points_gdf[attribute] == value for attribute,value in zip(attributes,comb)]
    final_mask = np.logical_and.reduce(individual_masks)
    return points_gdf[final_mask]

def k_most_cosine_similar(zero_points_gdf,non_zero_points_gdf,geo_settings,is_trans):
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    attributes = geo_settings[0]
    #for this we need to make all data numerical, combine first so consistent
    all_data = pd.concat([zero_points_gdf[attributes], non_zero_points_gdf[attributes]], axis=0)
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    ct = ColumnTransformer(transformers=[("cat", categorical_transformer, attributes)])
    transformed = ct.fit_transform(all_data)
    zero_atts = transformed[:len(zero_points_gdf)]
    non_zero_atts = transformed[len(zero_points_gdf):]
    cosine_sim_matrix = cosine_similarity(zero_atts, non_zero_atts)
    

def nearest(zero_points_gdf,non_zero_points_gdf,geo_settings):
    attributes = geo_settings[0]
    non_zero_att_combs,zero_att_combs = get_zero_non_zero_att_combs(non_zero_points_gdf,zero_points_gdf,attributes)
    sol_list = []
    for comb in non_zero_att_combs:
        if comb in zero_att_combs:
            zero_comb_gdf = filter_by_comb(zero_points_gdf,comb,attributes)
            non_zero_comb_gdf = filter_by_comb(non_zero_points_gdf,comb,attributes)
            sol_list.append(gpd.sjoin_nearest(zero_comb_gdf,non_zero_comb_gdf,how="left",distance_col="distance"))
    return pd.concat(sol_list, ignore_index=True)
            
def knn(zero_points_gdf,non_zero_points_gdf,geo_settings,is_trans):
    from sklearn.neighbors import NearestNeighbors
    attributes = geo_settings[0]
    n = geo_settings[2][1]
    non_zero_att_combs,zero_att_combs = get_zero_non_zero_att_combs(non_zero_points_gdf,zero_points_gdf,attributes)
    sol_list = []
    for comb in non_zero_att_combs:
        if comb in zero_att_combs:
            zero_comb_gdf = filter_by_comb(zero_points_gdf,comb,attributes)
            non_zero_comb_gdf = filter_by_comb(non_zero_points_gdf,comb,attributes)
            print(zero_comb_gdf.columns)
            if is_trans:
                zero_comb_gdf = zero_comb_gdf.set_index(["edge_from","edge_to"])
                non_zero_comb_gdf = non_zero_comb_gdf.set_index(["edge_from","edge_to"])
            else:
                zero_comb_gdf.set_index("edgeID")
                non_zero_comb_gdf.set_index("edgeID")
            coords1 = np.array(list(zip(zero_comb_gdf.geometry.x, zero_comb_gdf.geometry.y)))
            coords2 = np.array(list(zip(non_zero_comb_gdf.geometry.x, non_zero_comb_gdf.geometry.y)))
            #do nearest neighbor
            if len(coords2) < n:
                continue
            nn = NearestNeighbors(n_neighbors=n)
            nn.fit(coords2)
            distances, indices = nn.kneighbors(coords1)
            #to get edgeIDs back from indixes
            if is_trans:
                zero_comb_gdf_keys = zero_comb_gdf.index.to_frame(index=False)
                non_zero_comb_gdf_keys = non_zero_comb_gdf.index.to_frame(index=False)
            else:
                zero_comb_gdf_edgeIDs = zero_comb_gdf.index.to_numpy()
                non_zero_comb_gdf_edgeIDs = non_zero_comb_gdf.index.to_numpy()
            results = []
            for i, (dist_list, idx_list) in enumerate(zip(distances, indices)):
                if is_trans:
                    left_row = zero_comb_gdf_keys.iloc[i]
                else:
                    origin_edgeID = zero_comb_gdf_edgeIDs[i]
                for dist, idx in zip(dist_list, idx_list):
                    if is_trans:
                        right_row = non_zero_comb_gdf_keys.iloc[idx]
                        results.append({'edge_from_left': left_row["edge_from"],'edge_to_left': left_row["edge_to"],'edge_from_right': right_row["edge_from"],'edge_to_right': right_row["edge_to"],'distance': dist})
                    else:
                        neighbor_edgeID = non_zero_comb_gdf_edgeIDs[idx]
                        results.append({'edgeID_left': origin_edgeID,'edgeID_right': neighbor_edgeID,'distance': dist})
            sol_list.append(pd.DataFrame(results))
    return pd.concat(sol_list, ignore_index=True)
    
def dbscan(zero_points_gdf,non_zero_points_gdf,geo_settings,is_trans):
    from sklearn.cluster import DBSCAN
    from scipy.spatial import cKDTree
    attributes = geo_settings[0]
    eps = geo_settings[2][1]
    min_samp = geo_settings[2][2]
    print("Eps: ",eps)
    print("Min samples: ",min_samp)
    non_zero_att_combs,zero_att_combs = get_zero_non_zero_att_combs(non_zero_points_gdf,zero_points_gdf,attributes)
    sol_list = []
    for comb in non_zero_att_combs:
        if comb in zero_att_combs:
            zero_comb_gdf = filter_by_comb(zero_points_gdf,comb,attributes).copy()
            non_zero_comb_gdf = filter_by_comb(non_zero_points_gdf,comb,attributes).copy()
            coords = np.array(list(zip(non_zero_comb_gdf.geometry.x, non_zero_comb_gdf.geometry.y)))
            db = DBSCAN(eps=eps,min_samples=min_samp).fit(coords)
            non_zero_comb_gdf["cluster"] = db.labels_
            clustered_points = non_zero_comb_gdf[non_zero_comb_gdf["cluster"] != -1]
            if len(clustered_points) == 0:
                continue
            #cluster centroids
            clusters = clustered_points.groupby("cluster")
            cluster_centroids = clusters.geometry.apply(lambda g: g.unary_union.centroid)
            centroid_gdf = gpd.GeoDataFrame(cluster_centroids, geometry=cluster_centroids,crs=non_zero_comb_gdf.crs)
            centroid_gdf["cluster"] = centroid_gdf.index
            #closest cluster to each point in zero_comb_gdf
            zero_coords = np.array(list(zip(zero_comb_gdf.geometry.x, zero_comb_gdf.geometry.y)))
            centroid_coords = np.array(list(zip(centroid_gdf.geometry.x, centroid_gdf.geometry.y)))
            #get distances
            tree = cKDTree(centroid_coords)
            distances, indices = tree.query(zero_coords, k=1)
            #assign cluster
            zero_comb_gdf["matched_cluster"] = centroid_gdf.iloc[indices]["cluster"].values
            zero_comb_gdf["cluster_dist"] = distances
            #cluster id to edge id in clusters
            if is_trans:
                cluster_edgeid_map = clustered_points.groupby("cluster")[["edge_from","edge_to"]].apply(lambda df: list(zip(df["edge_from"],df["edge_to"]))).to_dict()
            else:
                cluster_edgeid_map = clustered_points.groupby("cluster")["edgeID"].apply(list).to_dict()
            #assign edgeids
            zero_comb_gdf["edgeID_right"] = zero_comb_gdf["matched_cluster"].apply(lambda cluster_id:cluster_edgeid_map.get(cluster_id,[]))
            sol_list.append(zero_comb_gdf)
    return pd.concat(sol_list, ignore_index=True)

def optics(zero_points_gdf,non_zero_points_gdf,geo_settings,is_trans):
    from sklearn.cluster import OPTICS
    from scipy.spatial import cKDTree
    attributes = geo_settings[0]
    #eps = geo_settings[2][1]
    #print("Eps: ",eps)
    min_samp = geo_settings[2][1]
    print("Min samples: ",min_samp)
    non_zero_att_combs,zero_att_combs = get_zero_non_zero_att_combs(non_zero_points_gdf,zero_points_gdf,attributes)
    sol_list = []
    for comb in non_zero_att_combs:
        if comb in zero_att_combs:
            zero_comb_gdf = filter_by_comb(zero_points_gdf,comb,attributes).copy()
            non_zero_comb_gdf = filter_by_comb(non_zero_points_gdf,comb,attributes).copy()
            coords = np.array(list(zip(non_zero_comb_gdf.geometry.x, non_zero_comb_gdf.geometry.y)))
            if len(coords) < min_samp:
                continue
            opt = OPTICS(min_samples=min_samp,max_eps=10000).fit(coords)
            non_zero_comb_gdf["cluster"] = opt.labels_
            clustered_points = non_zero_comb_gdf[non_zero_comb_gdf["cluster"] != -1]
            if len(clustered_points) == 0:
                continue
            #cluster centroids
            clusters = clustered_points.groupby("cluster")
            cluster_centroids = clusters.geometry.apply(lambda g: g.unary_union.centroid)
            centroid_gdf = gpd.GeoDataFrame(cluster_centroids, geometry=cluster_centroids,crs=non_zero_comb_gdf.crs)
            centroid_gdf["cluster"] = centroid_gdf.index
            #closest cluster to each point in zero_comb_gdf
            zero_coords = np.array(list(zip(zero_comb_gdf.geometry.x, zero_comb_gdf.geometry.y)))
            centroid_coords = np.array(list(zip(centroid_gdf.geometry.x, centroid_gdf.geometry.y)))
            #get distances
            tree = cKDTree(centroid_coords)
            distances, indices = tree.query(zero_coords, k=1)
            #assign cluster
            zero_comb_gdf["matched_cluster"] = centroid_gdf.iloc[indices]["cluster"].values
            zero_comb_gdf["cluster_dist"] = distances
            #cluster id to edge id in clusters
            if is_trans:
                cluster_edgeid_map = clustered_points.groupby("cluster")[["edge_from","edge_to"]].apply(lambda df: list(zip(df["edge_from"],df["edge_to"]))).to_dict()
            else:
                cluster_edgeid_map = clustered_points.groupby("cluster")["edgeID"].apply(list).to_dict()
            #assign edgeids
            zero_comb_gdf["edgeID_right"] = zero_comb_gdf["matched_cluster"].apply(lambda cluster_id:cluster_edgeid_map.get(cluster_id,[]))
            sol_list.append(zero_comb_gdf)
    return pd.concat(sol_list, ignore_index=True)



def get_geo_relatives_app(geo_data,freq_data,geo_settings,time_settings):
    zero_index_map,non_zero_index_map = split_zero_non_zero_behavs(freq_data)
    print("zero",len(zero_index_map))
    print("non zero",len(non_zero_index_map))
    edges_gdf = geo_data[1]
    centroids_gdf = get_centroids(edges_gdf)
    zero_points = app_behavs_to_geo(centroids_gdf,zero_index_map,time_settings)
    non_zero_points = app_behavs_to_geo(centroids_gdf,non_zero_index_map,time_settings)        
    methode = geo_settings[1]
    geo_relatives = defaultdict(lambda: ([],[]))
    if methode == "nearest":
        sol_gdf = nearest(zero_points,non_zero_points,geo_settings)
        for left,right,dist in zip(sol_gdf["edgeID_left"],sol_gdf["edgeID_right"], sol_gdf["distance"]):
            geo_relatives[left][0].append(right)
            geo_relatives[left][1].append(dist)
    elif methode == "knn":
        sol_df = knn(zero_points,non_zero_points,geo_settings,False)
        for left,right,dist in zip(sol_df["edgeID_left"],sol_df["edgeID_right"], sol_df["distance"]):
            geo_relatives[left][0].append(right)
            geo_relatives[left][1].append(dist)
    elif methode == "dbscan":
        print("dbscan")
        sol_df = dbscan(zero_points,non_zero_points,geo_settings,False)
        for left,right,dist in zip(sol_df["edgeID"],sol_df["edgeID_right"], sol_df["cluster_dist"]):
            geo_relatives[left][0].extend(right)
            geo_relatives[left][1].append(dist)
    elif methode == "optics":
        print("optics")
        sol_df = optics(zero_points,non_zero_points,geo_settings,False)
        for left,right,dist in zip(sol_df["edgeID"],sol_df["edgeID_right"], sol_df["cluster_dist"]):
            geo_relatives[left][0].extend(right)
            geo_relatives[left][1].append(dist)



    return geo_relatives
 
def trans_behavs_to_geo(geo_data,index_map,geo_settings,time_settings):
    is_time = time_settings[0]
    attributes = geo_settings[0]
    if is_time:
        transition_list = [t[1] for t in list(index_map.keys())]
    else:
        transition_list = list(index_map.keys())
    edges_gdf = geo_data[1]
    edges_gdf = edges_gdf.set_index("edgeID")
    centroids = []
    for edge_from, edge_to in transition_list:
        row_from = edges_gdf.loc[edge_from]
        row_to = edges_gdf.loc[edge_to]
        geom1 = row_from.geometry
        geom2 = row_to.geometry
        merged = linemerge([geom1,geom2])
        centroid = merged.centroid

        atts = { "edge_from":edge_from,"edge_to":edge_to,"geometry":centroid}

        for att in attributes:
            atts[f"{att}_1"] = row_from[att]
            atts[f"{att}_2"] = row_to[att]    
        centroids.append(atts)
    return gpd.GeoDataFrame(centroids,geometry="geometry",crs=edges_gdf.crs)


def get_geo_relatives_trans(geo_data,freq_data,geo_settings,time_settings):
    zero_index_map,non_zero_index_map = split_zero_non_zero_behavs(freq_data)
    zero_points = trans_behavs_to_geo(geo_data,zero_index_map,geo_settings,time_settings)    
    non_zero_points = trans_behavs_to_geo(geo_data,non_zero_index_map,geo_settings,time_settings)   
    methode = geo_settings[1]
    attributes = geo_settings[0]
    trans_atts = [att+"_1" for att in attributes ] + [att+"_2" for att in attributes]
    geo_settings[0] = trans_atts
    geo_relatives = defaultdict(lambda: ([],[]))
    if methode == "nearest":
        sol_gdf = nearest(zero_points,non_zero_points,geo_settings)
        for l_from, l_to, r_from, r_to, dist in zip(sol_gdf["edge_from_left"], sol_gdf["edge_to_left"],sol_gdf["edge_from_right"], sol_gdf["edge_to_right"],sol_gdf["distance"]):
            key = (l_from, l_to)
            geo_relatives[key][0].append((r_from, r_to))
            geo_relatives[key][1].append(dist)
    elif methode == "knn":
        sol_df = knn(zero_points,non_zero_points,geo_settings,True)
        for l_from, l_to, r_from, r_to, dist in zip(sol_df["edge_from_left"], sol_df["edge_to_left"],sol_df["edge_from_right"], sol_df["edge_to_right"],sol_df["distance"]):
            key = (l_from, l_to)
            geo_relatives[key][0].append((r_from, r_to))
            geo_relatives[key][1].append(dist)
    elif methode == "dbscan":
        sol_df = dbscan(zero_points,non_zero_points,geo_settings,True)
        for l_from, l_to, r, dist in zip(sol_df["edge_from"], sol_df["edge_to"],sol_df["edgeID_right"], sol_df["cluster_dist"]):
            key = (l_from, l_to)
            geo_relatives[key][0].extend(r)
            geo_relatives[key][1].append(dist)
    elif methode == "optics":
        sol_df = optics(zero_points,non_zero_points,geo_settings,True)
        for l_from, l_to, r, dist in zip(sol_df["edge_from"], sol_df["edge_to"],sol_df["edgeID_right"], sol_df["cluster_dist"]):
            key = (l_from, l_to)
            geo_relatives[key][0].extend(r)
            geo_relatives[key][1].append(dist)

    geo_settings[0] = attributes
    return geo_relatives

def relatives_to_behavs(relatives,interval):
    return { (interval,key):( [(interval,edges) for edges in value[0]],  value[1]) for key,value in relatives.items() }

def sum_up(geo_relatives,freq_data,geo_settings):
    adj_matrix = freq_data[0]
    behav_index_map = freq_data[1]
    methode = geo_settings[1]
    alpha = geo_settings[2][0]
    print("Alpha: ",alpha)
    if methode=="nearest":
        for target,source in geo_relatives.items():
            nearest = source[0][0]
            distance = source[1][0]
            #distance decay
            adj_matrix[:, behav_index_map[target]] = adj_matrix[:, behav_index_map[nearest]]*(1/(distance+1)**2)
    elif methode=="knn":
        for target,source in geo_relatives.items():
            neighbors = source[0]
            indexes = [behav_index_map.get(k, None) for k in neighbors]
            distances = np.array(source[1])
            weights = 1/(distances+1)**2
            adj_matrix[:, behav_index_map[target]] = (adj_matrix[:, indexes]*weights[np.newaxis,:]  ).sum(axis=1)/len(neighbors)
    elif methode=="dbscan":
        for target,source in geo_relatives.items():
            neighbors = source[0]
            distance = source[1][0]
            indexes = [behav_index_map.get(k, None) for k in neighbors]
            adj_matrix[:, behav_index_map[target]] = (adj_matrix[:, indexes].sum(axis=1)/len(neighbors))*(1/(distance+1)**2)*alpha
    elif methode=="optics":
        for target,source in geo_relatives.items():
            neighbors = source[0]
            distance = source[1][0]
            indexes = [behav_index_map.get(k, None) for k in neighbors]
            adj_matrix[:, behav_index_map[target]] = (adj_matrix[:, indexes].sum(axis=1)/len(neighbors))*(1/(distance+1)**2)*alpha
    return adj_matrix
            

def calc_geo_dependency(geo_data,freq_data,geo_settings,time_settings):
    app_behavs,trans_behavs = split_app_trans(geo_data,freq_data,time_settings)
    print("app",len(app_behavs[1]))
    print("trans",len(trans_behavs[1]))
    is_time = time_settings[0]
    if not is_time:
        intervals = 1
    else:
        intervals = time_settings[1]
    geo_relatives = {}
    for i in range(intervals):
        app_geo_relatives = get_geo_relatives_app(geo_data,get_interval_i_behavs(app_behavs,i,intervals),geo_settings,time_settings)
        trans_geo_relatives = get_geo_relatives_trans(geo_data,get_interval_i_behavs(trans_behavs,i,intervals),geo_settings,time_settings)
        if is_time:
            app_geo_relatives = relatives_to_behavs(app_geo_relatives,i)
            trans_geo_relatives = relatives_to_behavs(trans_geo_relatives,i)
        geo_relatives |= app_geo_relatives
        geo_relatives |= trans_geo_relatives
    adj_matrix = sum_up(geo_relatives,freq_data,geo_settings)
    return adj_matrix
