import os
from rwrBuildModel import create_model
from routing import create_route_graph,add_weight_to_dual,route,shortest_path
import random
import pandas as pd
import geopandas as gpd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
street_graph = os.path.join("..","gis","streetGraphs")
cleaned_travel = os.path.join("..","gis","cleanedTravelData")
matched_folder = os.path.join("..","gis","matchedTraces")
trip_info = os.path.join("..","gis","cleanedTravelData")
model_folder = os.path.join("..","gis","model")
results_folder = os.path.join("..","gis","results")

subfolder_name = ["philadelphia","beijing"]
utm_crs = ["EPSG:32618","EPSG:32650"]
crs_filename_append = [s.split(":")[1] for s in utm_crs]

def split_files(location_i,ratio,filtered_users=[],beijing_trans_label=None):
    path = os.path.join(matched_folder,subfolder_name[location_i])
    file_list = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(".shp")]
    
    if location_i==1:
        if beijing_trans_label!=None:
            trip_path = os.path.join(cleaned_travel,subfolder_name[location_i],"trips.csv")
            label_df = pd.read_csv(trip_path)
            all_users = set(label_df["UserId"].unique())
            if "carNone":
                correct_trans_users = set(label_df.loc[(label_df["label"] == "car") |label_df["label"].isna(), "UserId"].unique())
            else:
                correct_trans_users = set(label_df.loc[label_df["label"] == beijing_trans_label, "UserId"].unique())
            disappeared_users = list(all_users -correct_trans_users)
            if "carNone":
                trip_ids = list(label_df[(label_df["label"] == "car")| label_df["label"].isna()]["TripID"])
            else:
                trip_ids = list(label_df[label_df["label"] == beijing_trans_label]["TripID"])
            file_list = [os.path.join(path,   "tra_"+str(trip_id)+".shp"    ) for trip_id in trip_ids]
            print("Users gone cause of transport lable:",disappeared_users)
            filtered_users = filtered_users + disappeared_users
    
    if filtered_users!=[]:
        print(filtered_users)
        if location_i == 0:
            trip_df = gpd.read_file(os.path.join(cleaned_travel,subfolder_name[location_i],"tripsCleaned_32618.shp"))
        else:
            trip_df = pd.read_csv(os.path.join(cleaned_travel,subfolder_name[location_i],"trips.csv"))
        correct_ids = list(trip_df[~trip_df["UserId"].isin(filtered_users)]["TripID"])
        old_file_list = file_list
        file_list = []
        print("old_file_list len",len(old_file_list))
        for file in old_file_list:
            trip_id = int(os.path.splitext(os.path.basename(file))[0].split("_")[1])
            if trip_id in correct_ids:
                file_list.append(file)
        print("file_list len",len(file_list))



    random.shuffle(file_list)
    if not 0<=ratio<=1:
        print("ratio not between 0 and 1")
        return [],[]

    split = int(len(file_list)*ratio)
    training_data = file_list[:split]
    test_data = file_list[split:]
    return training_data,test_data,filtered_users

def get_network(location_i):
    path_edges = os.path.join(street_graph,subfolder_name[location_i],"edges_"+crs_filename_append[location_i]+".shp")
    path_nodes = os.path.join(street_graph,subfolder_name[location_i],"nodes_"+crs_filename_append[location_i]+".shp")

    edges_gdf = gpd.read_file(path_edges)
    nodes_gdf = gpd.read_file(path_nodes)
    return nodes_gdf,edges_gdf

def get_trip_info(location_i):
    if location_i == 0:
        return gpd.read_file(os.path.join(trip_info,subfolder_name[location_i],"tripsCleaned_32618.shp"))
    elif location_i == 1:
        return pd.read_csv(os.path.join(trip_info,subfolder_name[location_i],"trips.csv"))


def build_model(file_list,nodes_gdf,edges_gdf,trip_info,time,intervals,geo,features,methode,setting,filtered_users=[]):
    return create_model(file_list,nodes_gdf,edges_gdf,trip_info,time,intervals,geo,features,methode,setting,filtered_users)


def write_model(weight_matrix,user_index_map,behav_index_map,location_i,name):
    path = os.path.join(model_folder,subfolder_name[location_i])
    os.makedirs(path,exist_ok=True)
    np.save(os.path.join(path,name),weight_matrix)

    data_str_keys = {str(k): v for k, v in behav_index_map.items()}

    with open(os.path.join(path,name+"_users.json"), 'w') as f:
        json.dump(user_index_map, f, indent=4)

    with open(os.path.join(path,name+"_behav.json"), 'w') as f:
       json.dump(data_str_keys, f, indent=4)

def read_model(location_i,name):
    path = os.path.join(model_folder,subfolder_name[location_i])
    weight_matrix = np.load(os.path.join(path,name+".npy"),allow_pickle=True)

    with open(os.path.join(path,name+"_users.json"), 'r') as f:
        user_index_map = json.load(f)
        
    
    with open(os.path.join(path,name+"_behav.json"), 'r') as f:
        behav_index_map = json.load(f)

    user_index_map = {int(k): v for k,v in user_index_map.items()}

    behav_index_map = {eval(k): v for k, v in behav_index_map.items()}

    return [weight_matrix,user_index_map,behav_index_map]

def load_routable_graph(location_i):
    nodes_gdf,edges_gdf = get_network(location_i)
    return create_route_graph(edges_gdf)

def add_weights_for_user_time(G_dual,userID,interval,weight_matrix,user_index_map,behav_index_map):
    return add_weight_to_dual(G_dual,userID,interval,weight_matrix,user_index_map,behav_index_map)

def routing(G_dual_weight,G_prime,start_id,end_id,userID,interval,weight_matrix,user_index_map,behav_index_map,edges_gdf):
    return route(G_dual_weight,G_prime,start_id,end_id,userID,interval,weight_matrix,user_index_map,behav_index_map,edges_gdf)

def add_time_interval(trip_info,intervals,location_i):
    #if location_i == 0:
        trip_info["Start"] = trip_info["Start"].fillna("1970-01-01 00:00:00")
        trip_info["date_time_start"] = pd.to_datetime(trip_info["Start"],format='ISO8601')
        trip_info['seconds'] = trip_info['date_time_start'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)
        total_seconds = 24 * 3600
        trip_info['interval'] = (trip_info['seconds'] // (total_seconds / intervals)).astype(int)
        print("After time intervals", trip_info.head())
        return trip_info
    #elif location_i == 1:
    #    
    #    print("Not Implemented!!!")
    #    return None


def count_overlap(route1_gdf,route2_gdf):
    return len( set(route1_gdf["edgeID"]) & set(route2_gdf["edgeID"]))

def overlap_distance(route1_gdf,route2_gdf):
    overlap = set(route1_gdf["edgeID"]) & set(route2_gdf["edgeID"])
    if not overlap:
        return 0
    else:
        overlapping_edges = route1_gdf[route1_gdf["edgeID"].isin(overlap)]
        return overlapping_edges.geometry.length.sum()

#https://mathworld.wolfram.com/GiniCoefficient.html
def gini_coefficient(X):
    X_zero = X[X > 0]
    X = np.sort(X)
    n = len(X)
    i = np.arange(1, n+1)
    gini = np.sum((2 * i - n - 1) * X) / (n * np.sum(X))
    X_zero = np.sort(X_zero)
    n = len(X_zero)
    i = np.arange(1, n+1)
    gini_zero = np.sum((2 * i - n - 1) * X_zero) / (n * np.sum(X_zero))
    return gini,gini_zero

    


def calc_model_performance(location_i,weight_matrix,user_index_map,behav_index_map,trip_info,test_list,time,interval,G_prime,G_dual,edges_gdf):
    tripIDs = [int(os.path.basename(f).replace('tra_', '').replace('.shp', '')) for f in test_list ]
    trip_info = trip_info[trip_info['TripID'].isin(tripIDs)].copy()
    trip_info['UserId'] = trip_info['UserId'].astype(int)

    if time:
        trip_info = add_time_interval(trip_info,interval,location_i)
        trip_and_user = trip_info[trip_info['TripID'].isin(tripIDs)].set_index('TripID')[['UserId','interval']].apply(tuple, axis=1).to_dict()
        sorted_trip = dict(sorted(trip_and_user.items(), key=lambda item: (item[1][0],item[1][1])))
    else:
        trip_and_user = trip_info[trip_info['TripID'].isin(tripIDs)].set_index('TripID')['UserId'].to_dict()
        sorted_trip = dict(sorted(trip_and_user.items(), key=lambda item: item[1] ))
    precision_rec_seg = []
    precision_rec_dis = []
    precision_sd_seg = []
    precision_sd_dis = []
    recall_rec_seg = []
    recall_rec_dis = []
    recall_sd_seg = []
    recall_sd_dis = []
    previous_tupel = None #for each user-time combination, we have to add different weights to the graph, so we sort all combinations to not do that so often
    route_distribution_rec = np.zeros(len(edges_gdf)) 
    route_distribution_shortest = np.zeros(len(edges_gdf))
    #average length difference
    length_diff = []
    length_diff_sd = []
    for tripID,values in sorted_trip.items():
        t_interval = -1
        #print("User", user)
        if time:
            user = values[0]
            t_interval = values[1]
            #print("Time interval",t_interval)
        else:
            user = int(values)
        if values != previous_tupel:
            G_dual_weight = add_weights_for_user_time(G_dual,user,t_interval,weight_matrix,user_index_map,behav_index_map)
        route_gdf = gpd.read_file(os.path.join(matched_folder,subfolder_name[location_i],"tra_"+str(tripID)+".shp"))
        if len(route_gdf) == 0:
            continue
        start_node_id = route_gdf["u"].iloc[0]
        end_node_id = route_gdf["v"].iloc[-1]
        start_edge_id = route_gdf["edgeID"].iloc[0]
        end_edge_id = route_gdf["edgeID"].iloc[-1]
        rec_route_gdf = routing(G_dual_weight,G_prime,start_node_id,end_node_id,user,t_interval,weight_matrix,user_index_map,behav_index_map,edges_gdf)
        shortest_path_gdf = shortest_path(G_prime,start_edge_id,end_edge_id,edges_gdf)
        have_sd = True
        if not isinstance(shortest_path_gdf, pd.DataFrame):
            have_sd = False
        if len(rec_route_gdf) != 0:
            precision_rec_seg.append( count_overlap(rec_route_gdf,route_gdf)/len(rec_route_gdf) )
            precision_rec_dis.append( overlap_distance(rec_route_gdf,route_gdf)/rec_route_gdf.geometry.length.sum()  )
            route_distribution_rec[list(rec_route_gdf["edgeID"])] += 1
            length_diff.append(route_gdf.geometry.length.sum()-rec_route_gdf.geometry.length.sum())
        if have_sd and len(shortest_path_gdf) != 0:
            precision_sd_seg.append( count_overlap(shortest_path_gdf,route_gdf)/len(shortest_path_gdf) )
            precision_sd_dis.append( overlap_distance(shortest_path_gdf,route_gdf)/shortest_path_gdf.geometry.length.sum()  )
            route_distribution_shortest[list(shortest_path_gdf["edgeID"])] += 1
            length_diff_sd.append(route_gdf.geometry.length.sum()-shortest_path_gdf.geometry.length.sum())
        if len(route_gdf) != 0:
            recall_rec_seg.append( count_overlap(rec_route_gdf,route_gdf)/len(route_gdf) )
            recall_rec_dis.append( overlap_distance(rec_route_gdf,route_gdf)/route_gdf.geometry.length.sum()  )
            if have_sd:
                recall_sd_seg.append( count_overlap(shortest_path_gdf,route_gdf)/len(route_gdf) )
                recall_sd_dis.append(overlap_distance(shortest_path_gdf,route_gdf)/route_gdf.geometry.length.sum())
        #route_gdf.to_file("orgRoute.shp")
        #rec_route_gdf.to_file("recRoute.shp")
        #shortest_path_gdf.to_file("shortestRoute.shp")
        previous_tupel = values
        #print("-")
    precision_rec_seg_avg = sum(precision_rec_seg)/len(precision_rec_seg)
    precision_rec_dis_avg = sum(precision_rec_dis)/len(precision_rec_dis)
    precision_sd_seg_avg = sum(precision_sd_seg)/len(precision_sd_seg)
    precision_sd_dis_avg = sum(precision_sd_dis)/len(precision_sd_dis)
    recall_rec_seg_avg = sum(recall_rec_seg)/len(recall_rec_seg)
    recall_rec_dis_avg = sum(recall_rec_dis)/len(recall_rec_dis)
    recall_sd_seg_avg = sum(recall_sd_seg)/len(recall_sd_seg)
    recall_sd_dis_avg = sum(recall_sd_dis)/len(recall_sd_dis)
    route_distribution_rec_avg,route_distribution_rec_avg_zero =  gini_coefficient(route_distribution_rec)
    route_distribution_shortest_avg,route_distribution_shortest_avg_zero = gini_coefficient(route_distribution_shortest)
    length_diff_avg = sum(length_diff)/len(length_diff)
    length_diff_avg_sd = sum(length_diff_sd)/len(length_diff_sd)
    print("Model Precision Segments: ",precision_rec_seg_avg)
    print("Model Precision Distance: ",precision_rec_dis_avg)
    print("Shortest Path Precision Segments: ",precision_sd_seg_avg)
    print("Shortest Path Precision Distance: ",precision_sd_dis_avg)
    print("Model Recall Segments: ", recall_rec_seg_avg)
    print("Model Recall Distance: ", recall_rec_dis_avg)
    print("Shortest Path Recall Segements: ",recall_sd_seg_avg)
    print("Shortest Path Recall Distance: ",recall_sd_dis_avg)
    print("Rec. Route Distribution Gini: ",route_distribution_rec_avg)
    print("Rec. Route Distribution Gini without zero: ",route_distribution_rec_avg_zero)
    print("Shortest Path Distribution Gini: ",route_distribution_shortest_avg)
    print("Shortest Path Distribution Gini without zero: ",route_distribution_shortest_avg_zero)
    print("Average length difference between true route and recommended route (negative values: true route shorter on average): ",length_diff_avg)
    print("Average length difference between true route and shortest path route: ",length_diff_avg_sd)
    return [precision_rec_seg_avg,precision_rec_dis_avg,precision_sd_seg_avg,precision_sd_dis_avg,recall_rec_seg_avg,recall_rec_dis_avg,recall_sd_seg_avg,recall_sd_dis_avg,route_distribution_rec_avg,route_distribution_shortest_avg,route_distribution_rec_avg_zero,route_distribution_shortest_avg_zero,length_diff_avg,length_diff_avg_sd]

def write_results(location_i,results,combs,trials):
    path = os.path.join(results_folder,subfolder_name[location_i])
    os.makedirs(path,exist_ok=True)
    timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    res_file = os.path.join(path,timestamp+"results.txt")
    sd_prec_seg = []
    sd_prec_dis = []
    sd_recall_seg = []
    sd_recall_dis = []
    sd_dist_gini = []
    sd_dist_gini_zero = []
    avg_length_sd = []
    with open(res_file, 'w') as file:
        for comb,result in zip(combs,results):
            sd_prec_seg.append(result[2])
            sd_prec_dis.append(result[3])
            sd_recall_seg.append(result[6])
            sd_recall_dis.append(result[7])
            sd_dist_gini.append(result[9])
            sd_dist_gini_zero.append(result[11])
            avg_length_sd.append(result[13])
            file.write( "Time: "+ str(comb[0])+" Intervals: "+str(comb[1])+" Geo: "+str(comb[2])+" Attributs: "+str(comb[3])+" Methode: " +str(comb[4])+ " Settings: " + str(comb[5])+ "\n" )
            file.write("Precision Segments:"+str(result[0]) +" Precision Distance: "+ str(result[1]) +"\n")
            file.write("Recall Segments:"+str(result[4]) +" Recall Distance: "+ str(result[5]) +"\n")
            file.write("Route Distribution Gini Coef.: "+str(result[8])+"\n")
            file.write("Route Distribution Gini Coef. without Zero: "+str(result[10])+"\n")
            file.write("Average length difference between true route and recommended route (negative values: true route shorter on average):"+str(result[12])+"\n")
        sd_prec_seg_avg = sum(sd_prec_seg)/len(sd_prec_seg)
        sd_prec_dis_avg = sum(sd_prec_dis)/len(sd_prec_dis)
        sd_recall_seg_avg = sum(sd_recall_seg)/len(sd_recall_seg)
        sd_recall_dis_avg = sum(sd_recall_dis)/len(sd_recall_dis)
        sd_dist_gini_avg = sum(sd_dist_gini)/len(sd_dist_gini)
        sd_dist_gini_zero_avg = sum(sd_dist_gini_zero)/len(sd_dist_gini_zero)
        avg_length_sd_avg = sum(avg_length_sd)/len(avg_length_sd)
        file.write("Shortest Path Methode:\n")
        file.write("Precision Segments:"+str(sd_prec_seg_avg) +" Precision Distance: "+ str(sd_prec_dis_avg)+"\n" )
        file.write("Recall Segments:"+str(sd_recall_seg_avg) +" Recall Distance: "+ str(sd_recall_dis_avg)+"\n" )
        file.write("Shorest Path Distribution Gini Coef.: "+str(sd_dist_gini_avg)+"\n")
        file.write("Shorest Path Distribution Gini Coef. without zero: "+str(sd_dist_gini_zero_avg)+"\n")
        file.write("Average length difference between true route and shortest path route: "+str(avg_length_sd_avg)+"\n")
        file.write("Done "+str(trials) +" trials per combination.")



#time,interval,geo,atts,methode
#phil_combs = [ [False,0,False,[],""],[True,4,False,[],""],[False,0,True,["highway","bi_inf","green","water"],"nearest"],[True,4,True,["highway","bi_inf","green","water"],"nearest"],[True,12,True,["highway","bi_inf","green","water"],"nearest"]   ]




trials = 5
#setting: converg,p,(alpha,(knn)/(eps,min_cluster))

def template_no_time_geo(rwr_eps,p):
    return [False,0,False,[],"",[rwr_eps,p]]

def template_time(t_int):
    return [True,t_int,False,[],"",[5**(-5),0.5]]

def template_geo(atts,methode,alpha,s1=0,s2=0):
    return [False,0,True,atts,methode,[5**(-5),0.5,alpha,s1,s2]]

rwr_eps = [5**(-2),5**(-5),5**(-10)]
p = [0.1,0.5]
t_int = [2,6,10]
atts_phil = ["bi_inf","highway","green","water","surface","trafSignal","calming","slope_cat"]
knn = [3,5,10]
eps = [50,100,120,200]
minn = [2,5,7,10,15]

#phil_combs =( [  template_no_time_geo(rwr_e,p_i) for rwr_e in rwr_eps for p_i in p  ] 
            #+ [template_time(t) for t in t_int]
            #+ [ template_geo([att],"nearest",1) for att in atts_phil   ]
            #+ [ template_geo([att],"knn",1,3) for att in atts_phil    ]
            #+ [ template_geo(["bi_inf"],"knn",1,n) for n in knn    ]
            #+ [ template_geo([att],"dbscan",1,100,5) for att in atts_phil   ]
            #+ [ template_geo(["bi_inf"],"dbscan",1,epss,5) for epss in eps   ]
#            + [ template_geo(["bi_inf","highway","green","water","surface","trafSignal","calming","slope_cat"],"optics",1,minnn) for minnn in minn   ])


phil_combs =(   [template_no_time_geo(5**(-5),0.5)  ]
            +  [template_geo(atts_phil,"nearest",1) ]
            +  [template_geo(atts_phil,"knn",1,5) ]
            +  [template_geo(atts_phil,"knn",1,10) ]
            +  [template_geo(atts_phil,"dbscan",1,100,5)] 
            +  [template_geo(atts_phil,"dbscan",1,150,5)]
            +  [template_geo(atts_phil,"optics",1,5)]
            +  [template_time(t) for t in t_int])


atts_beij = ["highway","green","water","trafSignal"]
#t_int = [2,4,6]
#knn = [10,20]
#eps = [100]
#minn = [5]


#beij_combs = ([template_time(t) for t in t_int]
#            + [ template_geo([att],"nearest",1) for att in atts_beij   ]
#            + [ template_geo([att],"knn",1,3) for att in atts_beij    ]
#            + [ template_geo(["highway"],"knn",1,n) for n in knn    ]
#            + [ template_geo([att],"dbscan",1,100,5) for att in atts_beij   ]
#            + [ template_geo(["highway"],"dbscan",1,epss,5) for epss in eps   ]
#            + [ template_geo(["highway"],"dbscan",1,100,minnn) for minnn in minn   ]
#            + [  template_no_time_geo(rwr_e,p_i) for rwr_e in rwr_eps for p_i in p  ])

beij_combs =(  [ template_no_time_geo(5**(-5),0.5)]                                       
            +  [template_geo(atts_beij,"nearest",1)]
            +  [template_geo(atts_beij,"knn",1,5)]
            +  [template_geo(atts_beij,"knn",1,10)]
            +  [template_geo(atts_beij,"dbscan",1,100,5)] 
            +  [template_geo(atts_beij,"dbscan",1,150,5)]
            +  [template_geo(atts_beij,"optics",1,5) ]
            +  [template_time(t) for t in t_int])






filtered_users_phil = ["4"]+["47","43","120","119","267","66","148","147","144","84","83","79","379","378","225","366","92","89","364","363","362","360","358","350","35","370"]
filtered_users_beij = [17,153,128,68]+[21, 133, 149, 60, 172, 116, 27, 177, 31, 151, 137, 180, 48, 120, 49, 123, 178]


def statistics_getter():
    for location_i in range(0,2):
        nodes_gdf,edges_gdf = get_network(location_i)
        trip_info_df = get_trip_info(location_i)
        G_prime,G_dual = load_routable_graph(location_i)
        results = []
        if location_i == 0: 
            print("Philadelphia")
            combs = phil_combs
            filtered_users = filtered_users_phil
        elif location_i == 1:
            print("Beijing")
            combs = beij_combs
            filtered_users = filtered_users_beij
        initial_results_list = [np.zeros(14) for _ in combs]
        for trial in range(trials):
            #here filtered users
            train_list,test_list,filtered_users = split_files(location_i,0.8,filtered_users,"carNone")
            for i,comb in enumerate(combs):
                print("Trial: ",trial)
                print("Comb: ",comb)
                time = comb[0]
                interval = comb[1]
                geo = comb[2]
                atts = comb[3]
                methode = comb[4]
                setting = comb[5]
                weight_matrix,user_index_map,behav_index_map = build_model(train_list,nodes_gdf,edges_gdf,trip_info_df,time,interval,geo,atts,methode,setting,filtered_users)
                initial_results_list[i] += calc_model_performance(location_i,weight_matrix,user_index_map,behav_index_map,trip_info_df,test_list,time,interval,G_prime,G_dual,edges_gdf)
        for int_result in initial_results_list:
            results.append(list(int_result/trials))
        write_results(location_i,results,combs,trials)


def build_and_write_model(location_i,time,interval,geo,atts,methode,setting,file_name,filtered_users):
    train_list,_ = split_files(location_i,1,filtered_users)
    nodes_gdf,edges_gdf = get_network(location_i)
    trip_info = get_trip_info(location_i)
    weight_matrix,user_index_map,behav_index_map = build_model(train_list,nodes_gdf,edges_gdf,trip_info,time,interval,geo,atts,methode,setting,filtered_users)
    write_model(weight_matrix,user_index_map,behav_index_map,location_i,file_name)
    return [weight_matrix,user_index_map,behav_index_map]

def do_routing(model,location_i,user,t_interval,start_node_id,end_node_id):
    weight_matrix,user_index_map,behav_index_map = model
    G_prime,G_dual = load_routable_graph(location_i)
    nodes_gdf,edges_gdf = get_network(location_i)
    G_dual_weight = add_weights_for_user_time(G_dual,user,t_interval,weight_matrix,user_index_map,behav_index_map)
    rec_route_gdf = routing(G_dual_weight,G_prime,start_node_id,end_node_id,user,t_interval,weight_matrix,user_index_map,behav_index_map,edges_gdf)
    return rec_route_gdf

def plot_route(location_i,route_gdf):
    edges_gdf = get_network(location_i)[1]
    ax = edges_gdf.plot(color='lightgray', linewidth=0.5, figsize=(10, 10))
    route_gdf.plot(ax=ax, color='red', linewidth=2)
    plt.savefig("tempImg.png")
    plt.show()

def one_model():
    ###########building one model and writing to file
    #location_i = [0,1] 0 philadelphia, 1 bejing
    location_i = 1
    #time = True, False. If time intervals should be considered
    time = True
    #interval. If time==true, this should be number in [2,24]. only number divisible by 2 permitted. large time intervals are very memory intensive
    interval = 4
    #geo = True,False. if geo properties should be considered
    geo = True
    #attributes = [], list of attributes to consider. for philly: ["bi_inf","highway","green","water","surface","trafSignal","calming","slope_cat"]. for beijing: ["highway","trafSignal","green","water"]
    attributes = ["highway","green","water","trafSignal"]
    #methode = ["nearest","knn","dbscan","optics"] what methode to use for geo
    methode = "dbscan"
    #settings a list of at most 5 entries. first two have to be given always: cut off for rwr and p for rwr. third is alpha, weakening factor if geo. 4,5 for dbscan eps and min_size_cluster. 4 for knn k,4 min_size_cluster optics
    settings = [5**(-1),0.1,1,120,5]
    #filename
    filename = "simple_dbscan_model"
    #filtered_users, for phil stirng, beijing int
    filtered_users_phil = ["4"]
    filtered_users_beij = [17]
    filtered_users = filtered_users_beij

    model = build_and_write_model(location_i,time,interval,geo,attributes,methode,settings,filename,filtered_users)
    #can also read existing model
    #model = read_model(location_i,filename)
    ###############calculating a route
    #user id: the user the route should be caculated for.
    #t_interval: the time interval
    #start_node_id: start node in the street graph
    #end_node_id: end node in the street graph
    route_gdf = do_routing(model,location_i,1,1,3429740269,109902805)
    plot_route(location_i,route_gdf)


if __name__ == "__main__":
    ###########building one model and writing to file, doing one route, etc.
    #one_model()
    
    ############for getting statistics of many models
    statistics_getter()
