import os
import sys
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import time
import numpy as np
import json
from networkx.exception import NetworkXNoPath
from routing import create_route_graph,add_weight_to_dual,route,shortest_path_nodes
street_graph = os.path.join("..","gis","streetGraphs")
model_folder = os.path.join("..","gis","model")

subfolder_name = ["philadelphia","beijing"]
utm_crs = ["EPSG:32618","EPSG:32650"]
crs_filename_append = [s.split(":")[1] for s in utm_crs]



def cli_menu(title,options,less_linebreaks=False):
    if less_linebreaks:
        i = 0
    print(f"\n{title}")
    for i, opt in enumerate(options):
        if less_linebreaks:
            print(f"{i + 1}. {opt}",end="  ")
        else: 
            print(f"{i + 1}. {opt}")
        if less_linebreaks:
            if (i+1)%10==0:
                print("")
            i+=1
    if less_linebreaks:
        print("")
    while True:
        choice = input("Enter number: ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return (int(choice) - 1)
        print("Invalid choice, try again.")

def load_network(location_i):
    path_edges = os.path.join(street_graph,subfolder_name[location_i],"edges_"+crs_filename_append[location_i]+".shp")
    path_nodes = os.path.join(street_graph,subfolder_name[location_i],"nodes_"+crs_filename_append[location_i]+".shp")
    edges_gdf = gpd.read_file(path_edges)
    nodes_gdf = gpd.read_file(path_nodes)
    return nodes_gdf,edges_gdf

def list_models(location_i):
    model_path = os.path.join(model_folder,subfolder_name[location_i])
    if not os.path.exists(model_path):
        print("No models available, please calculate one.")
        sys.exit()
    npy_files = [f for f in os.listdir(model_path) if f.endswith(".npy")]
    models = []
    for npy in npy_files:
        name = npy.split(".")[0]
        if os.path.isfile(os.path.join(model_path,name+"_behav.json")) and os.path.isfile(os.path.join(model_path,name+"_users.json")):
            models.append(name)
    if len(models)==0:
        print("No models available, please calculate one.")
        sys.exit()
    else:
        return models

def load_model(location_i,name):
    path = os.path.join(model_folder,subfolder_name[location_i])
    weight_matrix = np.load(os.path.join(path,name+".npy"),allow_pickle=True)

    with open(os.path.join(path,name+"_users.json"), 'r') as f:
        user_index_map = json.load(f)


    with open(os.path.join(path,name+"_behav.json"), 'r') as f:
        behav_index_map = json.load(f)

    behav_index_map = {eval(k): v for k, v in behav_index_map.items()}

    return weight_matrix,user_index_map,behav_index_map


def find_nearest_node(click_point, nodes_gdf):
    distances = nodes_gdf.geometry.distance(click_point)
    return nodes_gdf.iloc[distances.idxmin()]



def click_selector(nodes_gdf,edges_gdf):
    print("Click to select start and end points, takes a second to load (close window to cancel).")
    fig, ax = plt.subplots()
    #nodes_gdf.plot(ax=ax, color='blue', markersize=5)
    edges_gdf.plot(ax=ax, linewidth=0.5, color='gray')
    plt.title("Right click to select start and end nodes",fontsize=20)
    
    clicked = []

    def on_click(event):
        if event.inaxes and event.button == 3:
            p = Point(event.xdata, event.ydata)
            nearest = find_nearest_node(p, nodes_gdf)
            clicked.append(nearest)
            print(f"Selected node: {nearest['nodeID']}")
            ax.plot(nearest.geometry.x, nearest.geometry.y, 'ro')
            ax.text(nearest.geometry.x, nearest.geometry.y, str(nearest['nodeID']), fontsize=16, color='black',path_effects=[path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
            fig.canvas.draw()
            if len(clicked) == 2:
                plt.pause(1)
                plt.close()

    fig.canvas.mpl_connect('button_press_event', on_click)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.show()
    return clicked if len(clicked) == 2 else (None, None)


def list_of_intervals(behav_index_map):
    intervals = []
    for i in behav_index_map.keys():
        if i[0] in intervals:
            break
        intervals.append(i[0])

    interval_hours = 24//len(intervals)

    intervals_as_text = []
    for i in range(len(intervals)):
        start_hour = i * interval_hours
        end_hour = (i + 1) * interval_hours
        intervals_as_text.append(f"{start_hour:02d}:00–{end_hour:02d}:00")

    return intervals,intervals_as_text

def display_route_window(route_gdf, edges_gdf,start_node,end_node,user_id,time_interval,shortest=False):
    fig, ax = plt.subplots()
    edges_gdf.plot(ax=ax, linewidth=0.5, color='gray')
    route_gdf.plot(ax=ax, color='red', linewidth=2)

    for node, color, label in [(start_node, 'blue', 'Start'), (end_node, 'green', 'End')]:
        x, y = node.geometry.x, node.geometry.y
        ax.plot(x, y, 'o', color=color)
        ax.text(x, y, f"{label}: {node['nodeID']}", fontsize=16, color="black",path_effects=[path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

    if shortest:
        plt.title(f"Calculated Route \n Shortest Path",fontsize=20)
    else:
        plt.title(f"Calculated Route \n UserID: {user_id} \n Time Interval: {time_interval}",fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    #block=False, the program will continue
    plt.show(block=False)


def load_network_and_model():
    location_i = cli_menu("Select a road network:",["Philadelphia","Beijing"])
    model_list = list_models(location_i)
    print("Loading routable graph, this may take a moment...")
    nodes_gdf,edges_gdf = load_network(location_i)
    G_prime,G_dual = create_route_graph(edges_gdf)
    print("Loaded!")
    model_i = cli_menu("Select a model:",model_list)
    print("Loading model, this may take a moment...")
    weight_matrix,user_index_map,behav_index_map = load_model(location_i,model_list[model_i])
    print("Loaded!")
    return nodes_gdf,edges_gdf,G_prime,G_dual,weight_matrix,user_index_map,behav_index_map

next_action = ['Change profile/time', 'Choose new start/end', 'Switch network/model', 'Exit']


def main():

    nodes_gdf,edges_gdf,G_prime,G_dual,weight_matrix,user_index_map,behav_index_map = load_network_and_model()

    while True:
        start_node,end_node = click_selector(nodes_gdf,edges_gdf)
        if start_node is None or end_node is None:
            print("Selection canceled.")
            break
        while True:
            start_node_id,end_node_id = start_node["nodeID"],end_node["nodeID"]
            user_list = list(user_index_map.keys())
            user_list.append("Shortest Path")
            user_id = user_list[cli_menu("Select a user:",user_list,True)]
            print("Selected user",user_id)
            if not user_id == "Shortest Path":
                interval_list,intervals_as_text = list_of_intervals(behav_index_map)
                interval_i = cli_menu("Select a time interval:",intervals_as_text)
                interval = interval_list[interval_i]
                interval_text = intervals_as_text[interval_i]
                print("Selected Time Interval",interval_text)
                G_dual_weight = add_weight_to_dual(G_dual,user_id,interval,weight_matrix,user_index_map,behav_index_map)
            try:
                if not user_id == "Shortest Path":
                    route_gdf = route(G_dual_weight,G_prime,start_node_id,end_node_id,user_id,interval,weight_matrix,user_index_map,behav_index_map,edges_gdf)
                    display_route_window(route_gdf,edges_gdf,start_node,end_node,user_id,interval_text)
                else:
                    route_gdf = shortest_path_nodes(G_prime,start_node,end_node,edges_gdf)
                    display_route_window(route_gdf,edges_gdf,start_node,end_node,"","",True)
            except NetworkXNoPath as e: 
                print(f"No path found between Node {start_node_id} and Node {end_node_id}",e)
            next_action_i = cli_menu("Chose next action:",next_action)
            if next_action_i == 0:
                continue
            elif next_action_i == 1:
                break
            elif next_action_i == 2:
                load_network_and_model()
                break
            else:
                sys.exit()


if __name__ == '__main__':
    main()

