import geopandas as gpd
import numpy as np
import os
import pandas as pd
from operator import itemgetter
import copy
from scipy.sparse import csr_matrix, csc_matrix, diags, issparse
from scipy.sparse.linalg import norm
import math
import matplotlib.pyplot as plt
import psutil
import gc
from shapely.ops import linemerge
from geoDependency import calc_geo_dependency

def create_adjacency_matrix_and_map(nodes_gdf,edges_gdf,trip_info,time=False,intervals=0,filtered_users=[]):
    users = trip_info.UserId.unique()
    users = [int(s) for s in users if s not in filtered_users]
    

    transitions = edges_gdf.merge(edges_gdf, left_on = 'v', right_on ='u', suffixes = ('_a','_b'))
    all_trans = list(zip(transitions.edgeID_a,transitions.edgeID_b))
    all_segs = list(edges_gdf.edgeID)
    all_behavs = all_segs+all_trans
    if time:
        app_behavior = [(t,b) for b in all_segs for t in range(intervals)]
        tra_behavior = [(t,b) for b in all_trans for t in range(intervals)]
        all_behavs = app_behavior+tra_behavior
    adjacency_matrix = np.zeros((len(users),len(all_behavs)))

    users_index_map = {k: v for v, k in enumerate(users)}
    behavs_index_map = {k: v for v, k in enumerate(all_behavs)}
    print(adjacency_matrix.shape)
    return adjacency_matrix,users_index_map,behavs_index_map

def add_intervals(trace_gdf,intervals):
    trace_gdf["date_time"] = pd.to_datetime(trace_gdf["date"] + " "+ trace_gdf['time'],format='ISO8601')
    trace_gdf['seconds'] = trace_gdf['date_time'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)
    total_seconds = 24 * 3600
    trace_gdf['interval'] = (trace_gdf['seconds'] // (total_seconds / intervals)).astype(int)
    return trace_gdf

def fill_matrix(file_list,trip_info,adj_matrix,user_index_map,behav_index_map,time,intervals):
    for file in file_list:
        tripID = int((os.path.basename(file).split("_")[1]).split(".")[0])
        trace_gdf = gpd.read_file(file)
        if len(trace_gdf)==0:
            continue
        userID = int(trip_info[trip_info["TripID"] == tripID].UserId.iloc[0])
        app_behavs = list(trace_gdf.edgeID)
        if time:
            trace_gdf = add_intervals(trace_gdf,intervals)
            app_behavs = list(zip(trace_gdf['interval'], trace_gdf['edgeID']))
        trans_behavs = list(zip(app_behavs[:-1],app_behavs[1:]))
        if time:
            trans_behavs = [(nest_t[1][0],(nest_t[0][1],nest_t[1][1])) for nest_t in trans_behavs ]
        all_behavs = app_behavs+trans_behavs
        try:
            behavIndexes = itemgetter(*all_behavs)(behav_index_map)
        #this happens if the trip did a transition that is not in the roadnetwork, could ne going the wrong way through a one way street
        #or the map matching just being wrong
        #we just discard
        except KeyError as e:
            missing_keys = [k for k in all_behavs if k not in behav_index_map]
            all_behavs = [e for e in all_behavs if e not in missing_keys]
            behavIndexes = itemgetter(*all_behavs)(behav_index_map)

        adj_matrix[user_index_map[userID],behavIndexes] += 1

    return adj_matrix


#https://stackoverflow.com/questions/8904694/how-to-normalize-a-2-dimensional-numpy-array-in-python-less-verbose
#if row is contains 0s, we leave these (so no divide by zero)
def row_norm(ad_matrix):
    #ad_matrix = ad_matrix.astype(float)
    row_sums = np.array(ad_matrix.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1
    # Create a diagonal matrix of 1 / row_sums
    inv_row_sums = diags(1 / row_sums)
    # Multiply from the left: D × A
    normalized = inv_row_sums @ ad_matrix
    return normalized
    #new_matrix = np.zeros_like(ad_matrix, dtype=float)
    #new_matrix[nonzero_rows] = ad_matrix[nonzero_rows] / row_sums[nonzero_rows, np.newaxis]
    #return new_matrix


def count_sparsity(A):
    sparsity = 1.0 - ( np.count_nonzero(A) / float(A.size) )
    return sparsity


def rwr(adj_matrix,p=0.1,convergence=5**(-10)):
    sparse_adj = csr_matrix(adj_matrix)
    ad_trans = sparse_adj.T.copy()
    q_1_2 = row_norm(sparse_adj)
    q_1_3 = row_norm(ad_trans)
    q_n_2 = q_1_2.copy()
    count = 0
    eps = 1000
    restart = p*q_1_2
    prod = q_1_2 @ q_1_3
    while eps>=convergence:
        print(count)
        prev = q_n_2
        q_n_2=(1-p)*(prod @ q_n_2)+restart
        eps = norm(q_n_2-prev)
        print(eps)
        count+=1
    return q_n_2.toarray()

def print_memory_usage(text):
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size (in bytes)
    mem_gb = mem_bytes / (1024 ** 3)
    print(text)
    print(f"Memory usage: {mem_gb:.2f} GB")


def circular_pairs(n, k):
    return [(i, (i + k) % n) for i in range(n)]

def temp_diff_hist(adj_matrix,behav_index_map,intervals):
    column_sums = adj_matrix.sum(axis=0).ravel()
    items = list(behav_index_map.items())
    interval_chunks = [dict(items[i:i+intervals]) for i in range(0, len(behav_index_map), intervals)]
    temp_intervals = intervals//2
    hist = {i:0 for i in range(0,temp_intervals+1)}
    print_memory_usage("2")
    for k in range(temp_intervals+1):
        if k == 0:
            hist[k] = sum( [math.comb(int(f),2) for f in column_sums ] ) 
        else:
            pairs = circular_pairs(intervals,k)
            for chunk in interval_chunks:
                indexes = list(chunk.values())
                index_pairs = [(indexes[pair[0]],indexes[pair[1]]) for pair in pairs ]
                for pair in index_pairs:
                    hist[k] += column_sums[pair[0]]*column_sums[pair[1]]
    for i in range(1,temp_intervals+1):
        half = hist[i]/2
        hist[i] = half
        hist[-i] = half
    return hist


#get list of lists, where each list is the time interval and it's previous, following: f.e. for n=12 7,8,9,10,11,0,1,2,3,4,5
def all_behav_time_combinations(n):
    base = list(range(n))
    tri_base = base+base+base
    return [tri_base[(i-(n//2-1)):(i+n//2)] for i in range(n,2*n)  ]

#for the behaviors, allways subset of behaviors of lenght intervals for same road segment or transition following each other
def split_index_into_time_interval_chunks(behav_index_map,intervals):
    all_indexes = list(behav_index_map.values())
    index_interval_chunks = [all_indexes[i:i+intervals] for i in range(0, len(behav_index_map), intervals)]
    return index_interval_chunks

def turn_combination_to_real_index(index_interval_chunks,combs):
    list_of_real_index_combs = []
    for index_chunk in index_interval_chunks:
        for comb in combs:
            list_of_real_index_combs.append([index_chunk[i] for i in comb])
    return list_of_real_index_combs


def calc_temp_dep(adj_matrix,hist_norm,behav_index_map,intervals):
    combs = all_behav_time_combinations(intervals)
    index_chunks =split_index_into_time_interval_chunks(behav_index_map,intervals)
    #all_index_combs = turn_combination_to_real_index(index_chunks,combs)
    mult_values = np.array(list(hist_norm.values()))[1:-1]
    counter = 0
    for index_chunk in index_chunks:
        for user_index in range(adj_matrix.shape[0]):
            if adj_matrix[user_index,index_chunk].sum() == 0:
                counter += intervals
                continue
            else:
                store = np.array(list(adj_matrix[user_index,index_chunk]))
                for comb in combs:
                    real_index = index_chunk[comb[intervals//2-1]]
                    comb_values = store[comb]
                    adj_matrix[user_index,real_index] = (mult_values*comb_values).sum()
        if index_chunk[0] % (12*1_000_00) == 0:
            print(index_chunk)
    return adj_matrix

def norm_and_shuffel_hist(hist):
    keys = list(hist.keys())
    split_index = keys.index(-1)
    before = list(hist.items())[:split_index]
    after = list(reversed(list(hist.items())[split_index:]))
    hist = dict(after+before)
    max_v = max(list(hist.values()))
    min_v = min(list(hist.values()))
    hist_norm = {k:((v-min_v)/(max_v-min_v)) for k,v in hist.items()  }
    return hist_norm

def add_temporal_dependency(adj_matrix,behav_index_map,intervals):
    hist = temp_diff_hist(adj_matrix,behav_index_map,intervals)
    print_memory_usage("3")
    hist_norm = norm_and_shuffel_hist(hist)
    adj_matrix = calc_temp_dep(adj_matrix,hist_norm,behav_index_map,intervals)
    plt.bar(list(hist_norm.keys()), list(hist_norm.values()))
    plt.savefig("./tempHist.png")
    return adj_matrix

########################################

def calc_app_prob(rwr_app,alpha):
    d = rwr_app.shape[1]
    row_sums = rwr_app.sum(axis=1)
    row_sums += alpha*d
    rwr_app += alpha
    result = rwr_app / row_sums[:,np.newaxis]
    return result

def calc_trans_prob(rwr_trans,behav_index_map,alpha,time):
    #getting rid of the time interval
    if not time: 
        transList = list(behav_index_map.keys())
    else:
        transList = [x[1] for x in list(behav_index_map.keys())]
    if not sorted(transList, key=lambda x: x[0]):
        print("Trans behavs not sorted by first edge")
        return None

    if len(transList) != rwr_trans.shape[1]:
        print("this should not happen")
        return None

    grouped_indices = {}
    for idx, (from_edge, _) in enumerate(transList):
        grouped_indices.setdefault(from_edge, []).append(idx)

    for key, indices in grouped_indices.items():
        chunk = rwr_trans[:,indices]
        d = chunk.shape[1]
        row_sums = chunk.sum(axis=1)
        row_sums += alpha*d
        chunk += alpha
        rwr_trans[:,indices] = chunk / row_sums[:,np.newaxis]
    return rwr_trans

def convert_to_probs(rwr_matrix,behav_index_map,intervals,split_point,time=False):
    alpha = 1e-20
    trans_i = split_point
    items = list(behav_index_map.items())
    app_matrix = rwr_matrix[:,:trans_i]
    trans_matrix = rwr_matrix[:,trans_i:]
    trans_index_map = dict(list(behav_index_map.items())[trans_i:])
    if not time:
        intervals = 1
    for i in range(intervals):
        app_matrix[:,i::intervals] = calc_app_prob(app_matrix[:,i::intervals],alpha)
        trans_matrix[:,i::intervals] = calc_trans_prob(trans_matrix[:,i::intervals],dict(list(trans_index_map.items())[i::intervals]),alpha,time)
    res = np.concatenate([app_matrix,trans_matrix],axis=1)
    return np.log(1/res)

##########################################################

def create_model(file_list,nodes_gdf,edges_gdf,trip_info,time=True,intervals=24,geo=True,features=["highway","bi_inf","green","water"],methode="nearest",setting=[5**(-10),0.1],filtered_users=[]):
    if time:
        if intervals>24 or intervals<2 or intervals%2!=0:
            print("Invalid interval")
            return None
    adj_matrix,user_index_map,behav_index_map = create_adjacency_matrix_and_map(nodes_gdf,edges_gdf,trip_info,time,intervals,filtered_users)
    adj_matrix = fill_matrix(file_list,trip_info,adj_matrix,user_index_map,behav_index_map,time,intervals)
    print_memory_usage("1")
    if time:
        print("Sparsity of matrix before temporal dependency: ",count_sparsity(adj_matrix))
        print("Memory used by matrix in GB:", adj_matrix.nbytes / (1024 ** 3))
        print("Number of non zero elements: ", np.count_nonzero(adj_matrix))

        print("Temporal Dependency")
        adj_matrix = add_temporal_dependency(adj_matrix,behav_index_map,intervals)
    print_memory_usage("5")
    print("Sparsity of matrix before rwr: ",count_sparsity(adj_matrix))
    print("Memory used by matrix in GB:", adj_matrix.nbytes / (1024 ** 3))
    print("Number of non zero elements: ", np.count_nonzero(adj_matrix))
    print("Do rwr")
    conv = setting[0]
    p = setting[1]
    print("Convergance rate:", conv)
    print("p: ",p)
    rwr_matrix = rwr(adj_matrix,p,conv)
    print_memory_usage("6")
    if geo:
        print("Sparsity of matrix before geographic dependency: ",count_sparsity(rwr_matrix))
        print("Memory used by matrix in GB:", rwr_matrix.nbytes / (1024 ** 3))
        print("Number of non zero elements: ", np.count_nonzero(rwr_matrix))
        print("Geographical Dependency")
        #rwr_matrix = add_geo_dependency(nodes_gdf,edges_gdf,rwr_matrix,behav_index_map,features,methode,time,intervals)
        rwr_matrix = calc_geo_dependency([nodes_gdf,edges_gdf],[rwr_matrix,behav_index_map],[features,methode,setting[2:]],[time,intervals] )
    print("Final Sparsity of matrix: ",count_sparsity(rwr_matrix))
    print("Number of non zero elements: ", np.count_nonzero(rwr_matrix))
    print("Memory used by matrix in GB:", rwr_matrix.nbytes / (1024 ** 3))
    if not time:
        interval = 1
    else:
        interval = intervals
    split_point = len(edges_gdf)*interval
    final_matrix = convert_to_probs(rwr_matrix,behav_index_map,intervals,split_point,time)
    print_memory_usage("After probs.")
    return final_matrix,user_index_map,behav_index_map
    


