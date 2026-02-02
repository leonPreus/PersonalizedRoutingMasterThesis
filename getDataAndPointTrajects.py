import osmnx as ox
import geopandas as gpd
import os
import numpy as np
import pandas as pd
from shapely import Point
from shapely import Polygon
import ast
import rasterio
from shapely.geometry import LineString
from rasterio.sample import sample_gen

travel_data = os.path.join("..","gis","originalTravelData")
boundary = os.path.join("..","gis","clip_shapes")
street_graph = os.path.join("..","gis","streetGraphs")
dest = os.path.join("..","gis","pointTraces")
cleaned_travel = os.path.join("..","gis","cleanedTravelData")
dem_folder = os.path.join("..","gis","DEMs")
other_data = os.path.join("..","gis","otherData")

subfolder_name = ["philadelphia","beijing"]
places = ["Philadelphia, USA","Beijing, China"]
#utm 18, utm 50
utm_crs = ["EPSG:32618","EPSG:32650"]
crs_filename_append = [s.split(":")[1] for s in utm_crs]
network_type = ['bike','drive']

def get_boundary(location_i):
    folder = os.path.join(boundary,subfolder_name[location_i])
    file_name = "boundary_"+crs_filename_append[location_i]+".shp"
    file_loc = os.path.join(folder,file_name)
    if not os.path.isfile(file_loc):
        raise FileNotFoundError("Boundary shape file not found. Should have name boundary_crs.shp ")
    boundary_gdf = gpd.read_file(file_loc)
    if boundary_gdf.crs != utm_crs[location_i]:
        boundary_gdf = boundary_gdf.to_crs(utm_crs[location_i])
        print("Reprojecting boundary to crs ",utm_crs[location_i])
    return boundary_gdf

def get_highway_feature_gdf(location_i,boundary_gdf):
    tags = {"highway": True}
    if location_i == 0:
        gdf = ox.features_from_place(places[location_i], tags=tags)
    elif location_i == 1:
        polygon = boundary_gdf.to_crs("EPSG:4326").geometry.iloc[0]
        gdf = ox.features_from_polygon(polygon, tags=tags)
    gdf = gdf.to_crs(utm_crs[location_i])
    return gdf

def bi_inf(row,cycleway_columns):
    if row["highway"] == "cycleway":
        return True
    #negative tags are really rare in osm anyway
    elif pd.notnull(row["cycleway"]) and row["cycleway"] != "no" :
        return True
    elif row[[col for col in cycleway_columns if col != "cycleway"]].notna().any():
        return True
    else:
        return False

    #return row["highway"] == "cycleway" or (pd.notnull(row["cycleway"]) and row["cycleway"] != "no" )
    

def add_bicycle_info(gdf,feature_gdf):
    cycleway_columns = [col for col in feature_gdf.columns if col.startswith('cycleway') and not col.startswith('cycleway:crossing')]
    feature_gdf = feature_gdf[cycleway_columns]
    result = gdf.merge(feature_gdf,left_on="osmid",right_index=True,how="left")
    result["bi_inf"] = result.apply(lambda x:bi_inf(x,cycleway_columns),axis=1)
    result = result.drop(cycleway_columns,axis=1)
    return result

def get_nature_gdf(location_i,boundary_gdf):
    tags = {"landuse": ["forest","grass","cemetery","farmland","meadow"],"natural":["grassland","wood",
            "scrub","wetland","heath"],"leisure":["garden","golf_course","park","playground"]}
    if location_i == 0:
        gdf = ox.features_from_place(places[location_i], tags=tags)
    elif location_i == 1:
        polygon = boundary_gdf.to_crs("EPSG:4326").geometry.iloc[0]
        gdf = ox.features_from_polygon(polygon, tags=tags)
    print("Download of nature features finished")
    gdf = gdf[gdf.geom_type.isin(['Polygon', 'MultiPolygon'])]
    gdf = gdf[["geometry"]]
    gdf = gdf.to_crs(utm_crs[location_i])
    gdf = gdf.dissolve()
    gdf["green"] = pd.Series([True])
    return gdf

def get_water_gdf(location_i,boundary_gdf):
    tags = {"natural": "water","waterway":["river","stream","canal"]}
    if location_i == 0:
        gdf = ox.features_from_place(places[location_i], tags=tags)
    elif location_i == 1:
        polygon = boundary_gdf.to_crs("EPSG:4326").geometry.iloc[0]
        gdf = ox.features_from_polygon(polygon, tags=tags)
    print("Download of water features finished.")
    gdf = gdf.to_crs(utm_crs[location_i])
    gdf_area = gdf[gdf.geom_type.isin(['Polygon', 'MultiPolygon'])]
    gdf_line = gdf[gdf.geom_type.isin(['LineString', 'MultiLineString'])]
    gdf_area = gdf_area[["geometry"]]
    gdf_line = gdf_line[["geometry"]]
    gdf_buffered = gdf_line.copy()
    gdf_buffered["geometry"] = gdf_buffered.buffer(10)
    gdf_comb = pd.concat([gdf_area,gdf_buffered],axis=0,ignore_index=True)
    gdf_comb = gdf_comb.dissolve()
    gdf_comb["water"] = pd.Series([True])
    return gdf_comb

#https://gis.stackexchange.com/questions/269243/creating-polygon-grid-using-geopandas
def create_grid_gdf(boundary_gdf):
    xmin,ymin,xmax,ymax = boundary_gdf.total_bounds
    step = 300
    
    cols = list(np.arange(xmin, xmax + step, step))
    rows = list(np.arange(ymin, ymax + step, step))

    polygons = []
    ids = []
    i = 0
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(Polygon([(x,y), (x+step, y), (x+step, y+step), (x, y+step)]))
            ids.append(i)
            i+=1
    grid_gdf = gpd.GeoDataFrame({"id":ids,'geometry':polygons},crs=boundary_gdf.crs)
    return grid_gdf

def get_feature_cells(grid_gdf,feature_gdf,feature_name):
    from shapely.geometry import box
    minx,miny,maxx,maxy = grid_gdf.total_bounds
    #some large features from OSM extend the grid a lot, easier to clip them first
    grid_bbox = gpd.GeoDataFrame(geometry=[box(minx,miny,maxx,maxy)],crs=grid_gdf.crs)
    feature_gdf = gpd.clip(feature_gdf,grid_bbox)
    cut_off = 0.3
    grid_gdf["area_grid"] = grid_gdf.area
    print("Overlaying grid with features, this may take a while")
    feature_gdf["geometry"] = feature_gdf.geometry.simplify(tolerance=40)
    gdf_joined = gpd.overlay(grid_gdf,feature_gdf,how="union")
    print("Done")
    gdf_joined["area_joined"] = gdf_joined.area
    gdf_joined['percentage'] = (gdf_joined['area_joined'] / gdf_joined['area_grid'])
    #since the feature polygon is just one dissloved polygon, each grid cell will now consists of at most two polygons
    #one where feature=True and one where feature=Null. The feature=True will have as percentage the area of the grid cell that contains that feature
    gdf_only_feature_part = gdf_joined[gdf_joined[feature_name] == True]
    if not gdf_only_feature_part["id"].is_unique:
        print("This should/can not happen.")
    df_only_feature_perc = gdf_only_feature_part.drop(columns="geometry")
    grid_with_feature_perc = grid_gdf.merge(df_only_feature_perc,on="id",how="left")
    #fill Null values with 0 (no part of feature in that cell
    grid_with_feature_perc["percentage"] = grid_with_feature_perc["percentage"].fillna(0)
    feature_cells = grid_with_feature_perc[grid_with_feature_perc["percentage"] >= cut_off]
    return feature_cells[[feature_name,"geometry"]]


def join_value_to_network(gdf,cells,feature_name):
    joined = gpd.sjoin(gdf,cells,how="left",predicate="intersects")
    grouped = joined.groupby("edgeID")[feature_name].any()
    gdf = gdf.merge(grouped,on="edgeID",how="left")
    return gdf


def add_landscape_water_info(gdf,nature_feature_gdf,water_feature_gdf,boundary_gdf):
    grid_gdf = create_grid_gdf(boundary_gdf)

    green_cells = get_feature_cells(grid_gdf,nature_feature_gdf,"green")
    water_cells = get_feature_cells(grid_gdf.copy(deep=True),water_feature_gdf,"water")

    gdf = join_value_to_network(gdf,green_cells,"green")
    gdf = join_value_to_network(gdf,water_cells,"water")

    return gdf

surface_mapping = {
    # paved_smooth
    'asphalt': 'paved_smooth',
    'concrete': 'paved_smooth',
    'paving_stones': 'paved_smooth',
    'wood': 'paved_smooth',
    'concrete:plates': 'paved_smooth',
    'bricks': 'paved_smooth',
    'concrete:lanes': 'paved_smooth',
    'brick': 'paved_smooth',
    'metal': 'paved_smooth',
    # paved_not_smooth
    'paved': 'paved_not_smooth',
    'sett': 'paved_not_smooth',
    'cobblestone': 'paved_not_smooth',
    'unhewn_cobblestone': 'paved_not_smooth',
    # unpaved
    'ground': 'unpaved',
    'dirt': 'unpaved',
    'unpaved': 'unpaved',
    'compacted': 'unpaved',
    'grass': 'unpaved',
    'gravel': 'unpaved',
    'fine_gravel': 'unpaved',
    'pebblestone': 'unpaved',
    'rock': 'unpaved',
    'sand': 'unpaved',
    'grass_paver': 'unpaved',
    'bing': 'unpaved'
}

state_road_map = {
    "99": "paved_smooth",
    "98": "paved_smooth",
    "80": "paved_not_smooth",
    "75": "paved_smooth",
    "74": "paved_smooth",
    "72": "paved_smooth",
    "71": "paved_smooth",
    "62": "paved_smooth",
    "61": "paved_smooth",
    "52": "paved_smooth",
    "51": "paved_smooth"
}

def classify_local(row):
    road_type_column = ["UNIMPROVED","GRAVEL_MIL","SEAL_COATE","BITUMINOUS","BRICK_MILE","CONCRETE_M"]
    if row[road_type_column].isnull().all():
        return "unknown"
    max_col = row[road_type_column].idxmax()
    if max_col in ["UNIMPROVED", "GRAVEL_MIL"]:
        return "unpaved"
    elif max_col == "BRICK_MIL":
        return "paved_not_smooth"
    elif max_col in ["SEAL_COATE", "BITUMINOUS", "CONCRETE_M"]:
        return "paved_smooth"
    else:
        return "unknown"

def enhance_surface(gdf,boundary_gdf):
    state_road_gdf = gpd.read_file(os.path.join(other_data,"Pennsylvania_State_Roads","RMSSEG_(State_Roads).shp"))
    local_road_gdf = gpd.read_file(os.path.join(other_data,"Pennsylvania_Local_Roads","Pennsylvania_Local_Roads.shp"))
    state_road_gdf = state_road_gdf.to_crs("EPSG:32618")
    local_road_gdf = local_road_gdf.to_crs("EPSG:32618")
    state_road_gdf = gpd.clip(state_road_gdf,boundary_gdf)
    local_road_gdf = gpd.clip(local_road_gdf,boundary_gdf)
    state_road_gdf = state_road_gdf[["OBJECTID","SURF_TYPE","geometry"]]
    state_road_gdf["surf_ogd"] = state_road_gdf["SURF_TYPE"].map(state_road_map).fillna("unknown")
    local_road_gdf = local_road_gdf[["OBJECTID","SEGMENT_LE","UNIMPROVED","GRAVEL_MIL","SEAL_COATE","BITUMINOUS","BRICK_MILE","CONCRETE_M","geometry"]]
    local_road_gdf["surf_ogd"] = local_road_gdf.apply(classify_local,axis=1)
    state_road_gdf = state_road_gdf[["surf_ogd","geometry"]]
    local_road_gdf = local_road_gdf[["surf_ogd","geometry"]]
    all_roads_gdf = pd.concat([state_road_gdf,local_road_gdf])
    print("Before sjoin ",len(gdf))
    #sjoin nearest can result in duplication of input if multiple records are same distance
    gdf = gpd.sjoin_nearest(gdf, all_roads_gdf, how="left", max_distance=1)
    gdf = gdf.drop_duplicates(subset="edgeID", keep="first")
    #duplicates = gdf[gdf.duplicated(subset="edgeID", keep=False)]
    #print(duplicates)
    print("After sjoin ",len(gdf))
    gdf["surf_ogd"] = gdf["surf_ogd"].fillna("unknown")
    gdf = gdf.drop(columns=["index_right"])
    assign_cond = (gdf["surface"] == "unknown") & (gdf["surf_ogd"] != "unknown")
    gdf.loc[assign_cond,"surface"] = gdf.loc[assign_cond,"surf_ogd"]
    #in these cases surface from OSM seems to be better
    conflict_cond = ((gdf["surface"] != "unknown") &(gdf["surf_ogd"] != "unknown") &(gdf["surface"] != gdf["surf_ogd"]))
    gdf["conflict"] = conflict_cond
    return gdf

def add_surface_info(gdf,feature_gdf,boundary_gdf,location_i):
    surface_gdf = feature_gdf[["surface"]]
    gdf = gdf.merge(surface_gdf,left_on="osmid",right_index=True,how="left")
    gdf["surface"] = gdf["surface"].map(surface_mapping).fillna('unknown')
    if location_i == 0:
        gdf = enhance_surface(gdf,boundary_gdf)
    return gdf

def add_traffic_signal_info(gdf,feature_gdf):
    buffer_size = 10
    traffic_signals = feature_gdf[feature_gdf["highway"] == "traffic_signals"][["highway","geometry"]]
    gdf["buffer"] = gdf.geometry.buffer(buffer_size)
    gdf_buffers = gdf[["edgeID","buffer"]].copy()
    gdf_buffers = gdf_buffers.set_geometry("buffer")
    join_result = gpd.sjoin(gdf_buffers,traffic_signals, predicate="contains",how="left")
    matched_edgeIDs = join_result[~join_result["highway"].isna()]["edgeID"].unique()
    gdf["trafSignal"] = gdf["edgeID"].isin(matched_edgeIDs)
    gdf = gdf.drop(columns="buffer")
    return gdf

def get_elevation(point,dem):
    coords = [(point.x,point.y)]
    return list(sample_gen(dem,coords))[0][0]

def get_slope(x,dem):
    if isinstance(x,LineString):
        start = x.coords[0]
        end = x.coords[-1]
        from shapely.geometry import Point
        p_start = Point(start)
        p_end = Point(end)
        elev_start = get_elevation(p_start,dem)
        elev_end = get_elevation(p_end,dem)
        return ((elev_start-elev_end)/x.length)*100
    else:
        print("Bad Geometry")
        return None

def add_slope(gdf,location_i):
    dem_path = os.path.join(dem_folder,subfolder_name[location_i],"mosaic_reprojected.tif")
    dem = rasterio.open(dem_path)
    gdf["slope"] = gdf["geometry"].apply(lambda x:get_slope(x,dem))
    gdf["slope_cat"] =  pd.cut(gdf['slope'],bins=[-float('inf'), -7,-3,3,7, float('inf')],
    labels=['downhill_steep',"downhill", 'flat', 'uphill', 'uphill_steep'])
    return gdf

def add_fix_maxspeed(gdf,boundary_gdf,location_i):
    gdf["maxspeed"] = gdf["maxspeed"].str.replace("mph","",case=False).str.strip()
    gdf["maxspeed"] = gdf["maxspeed"].fillna("unknown")
    full_road_gdf = gpd.read_file(os.path.join(other_data,"withSpeed.shp"))
    full_road_gdf = full_road_gdf.to_crs(utm_crs[location_i])
    full_road_gdf = gpd.clip(full_road_gdf,boundary_gdf)
    full_road_gdf = full_road_gdf[["SPEED_LIMI","geometry"]]
    gdf = gpd.sjoin_nearest(gdf, full_road_gdf, how="left", max_distance=1)
    gdf = gdf.drop_duplicates(subset="edgeID", keep="first")
    gdf["SPEED_LIMI"] = gdf["SPEED_LIMI"].fillna(0)
    gdf["SPEED_LIMI"] = gdf["SPEED_LIMI"].astype(int).astype(str)
    gdf["SPEED_LIMI"] = gdf["SPEED_LIMI"].replace('0',"unknown")
    gdf = gdf.drop(columns=["index_right"])
    assign_cond = (gdf["maxspeed"] == "unknown") & (gdf["SPEED_LIMI"] != "unknown")
    gdf.loc[assign_cond,"maxspeed"] = gdf.loc[assign_cond,"SPEED_LIMI"]
    #in these cases speed from OSM seems to be better
    conflict_cond = ((gdf["maxspeed"] != "unknown") &(gdf["SPEED_LIMI"] != "unknown") &(gdf["maxspeed"] != gdf["SPEED_LIMI"]))
    gdf["conf_speed"] = conflict_cond
    return gdf

def add_bumps(gdf,feature_gdf_way,feature_gdf_node,location_i):
    calm_way_gdf = feature_gdf_way[feature_gdf_way["traffic_calming"].notna()]
    calm_node_gdf = feature_gdf_node[feature_gdf_node["traffic_calming"].notna()]
    tags = {"traffic_calming": True}
    calming_osm_gdf = ox.features_from_place(places[location_i], tags=tags)
    calming_osm_gdf = calming_osm_gdf.to_crs(utm_crs[location_i])
    print("Download of traffic calming features finished.")
    calming_ogd_gdf = gpd.read_file(os.path.join(other_data,"traffic_calming_devices","traffic_calming_devices.shp"))
    calming_ogd_gdf = calming_ogd_gdf.to_crs(utm_crs[location_i])
    node_gdf1 = calm_node_gdf[["geometry"]]
    node_gdf2 = calming_osm_gdf[["geometry"]]
    node_gdf3 = calming_ogd_gdf[["geometry"]]
    gdf_all_bumpers = gpd.GeoDataFrame(pd.concat([node_gdf1, node_gdf2, node_gdf3], ignore_index=True),geometry='geometry',crs=node_gdf1.crs)
    gdf_all_bumpers["highway"] = "placeholder"

    buffer_size = 10
    gdf["buffer"] = gdf.geometry.buffer(buffer_size)
    gdf_buffers = gdf[["edgeID","buffer"]].copy()
    gdf_buffers = gdf_buffers.set_geometry("buffer")
    join_result = gpd.sjoin(gdf_buffers,gdf_all_bumpers, predicate="contains",how="left")
    matched_edgeIDs = join_result[~join_result["highway"].isna()]["edgeID"].unique()
    gdf["bumpers"] = gdf["edgeID"].isin(matched_edgeIDs)
    gdf = gdf.drop(columns="buffer")
    

    return gdf

def add_calming(gdf,boundary_gdf,feature_gdf_way,feature_gdf_node,location_i):
    gdf = add_fix_maxspeed(gdf,boundary_gdf,location_i)
    gdf = add_bumps(gdf,feature_gdf_way,feature_gdf_node,location_i)
    gdf["maxspeed"] = gdf["maxspeed"].replace("unknown","100").astype(int)
    gdf["calming"] = (gdf["maxspeed"] < 25) | (gdf["bumpers"] == True) | gdf["highway"].isin(["path","cycleway","track","pedestrian","living_street"])
    gdf["maxspeed"] = gdf["maxspeed"].astype(str).replace("100","unknown")
    return gdf

def add_geo_info(gdf,location_i,boundary_gdf,feature_gdf):
    feature_gdf_node = feature_gdf.xs("node",level="element")
    feature_gdf_way = feature_gdf.xs('way', level='element')
    #clean highway column
    for road_type in ["motorway","trunk","primary","secondary","tertiary"]:
        gdf["highway"] = gdf["highway"].replace(road_type+"_link", road_type)
    #remove some useless classifications
    if location_i == 1:
        gdf["highway"] = gdf["highway"].replace(["scramble","road","busway","rest_area","escape","disused"],"unclassified")
    #add bicycle info
    if location_i == 0:
        gdf = add_bicycle_info(gdf,feature_gdf_way)
    print("2",len(gdf))
    #add surface info
    gdf = add_surface_info(gdf,feature_gdf_way,boundary_gdf,location_i)
    print("2.1",len(gdf))
    #add traffic signals info
    gdf = add_traffic_signal_info(gdf,feature_gdf_node)
    print("2.2",len(gdf))
    if location_i == 0:
        gdf = add_slope(gdf,location_i)
    print("2.3",len(gdf))
    if location_i == 0:
        gdf = add_calming(gdf,boundary_gdf,feature_gdf_way,feature_gdf_node,location_i)
    #add landscape info
    print("3",len(gdf))
    nature_feature_gdf = get_nature_gdf(location_i,boundary_gdf)
    water_feature_gdf = get_water_gdf(location_i,boundary_gdf)
    gdf = add_landscape_water_info(gdf,nature_feature_gdf,water_feature_gdf,boundary_gdf)
    if location_i == 0:
        gdf = gdf.drop(columns=["lanes","surf_ogd","conflict","slope","SPEED_LIMI","conf_speed","bumpers"])
    return gdf

def largest_overlap(x,feature_gdf,other_columns):
    osmids = x["osmid"]
    ways = feature_gdf[feature_gdf.index.isin(osmids)]
    ways["overlap"] = ways.geometry.intersection(x.geometry).length
    if len(ways)==0:
        #this happens rarely
        x["osmid"] = osmids[0]
        for col in other_columns:
            if isinstance(x[col],list):
                x[col] = x[col][0]
        return x
    max_overlap_idx = ways["overlap"].idxmax()
    x["osmid"] = max_overlap_idx
    max_overlap_way = ways.loc[max_overlap_idx]
    for col in other_columns:
        #this happens rarely
        if isinstance(max_overlap_way[col], (list, pd.Series, gpd.GeoSeries, np.ndarray)):
            x[col] = max_overlap_way[col].iloc[0]
            #print(ways)
            #print(max_overlap_idx)
            #print(max_overlap_way[col])
        else:
            x[col] = max_overlap_way[col]
    return x

def fix_listed_columns(edges_gdf,other_columns,location_i,feature_gdf):
    list_mask = edges_gdf['osmid'].apply(lambda x: isinstance(x, list))
    feature_gdf = feature_gdf.xs('way', level='element')
    edges_gdf[list_mask] = edges_gdf[list_mask].apply(lambda x: largest_overlap(x, feature_gdf, other_columns),axis=1)
    return edges_gdf

def replace_start_end_LineString(row,new_geom,do_start):
    coords = list(row.geometry.coords)
    if do_start:
        coords[0] = (new_geom.x,new_geom.y)
    else:
        coords[-1] = (new_geom.x,new_geom.y)
    return LineString(coords)

#this only happens for 2 nodes in Beijing. Nodes should not overlap at all. makes problems later if not fixed
def shift_overlapping_nodes(nodes_gdf,edges_gdf):
    dx = 1
    nodes_gdf["x"] = nodes_gdf.geometry.x
    nodes_gdf["y"] = nodes_gdf.geometry.y
    groups = nodes_gdf.groupby(["x","y"]).groups
    for (x,y),idxs in groups.items():
        if len(idxs) == 2:
            #nodeID_0 = nodes_gdf.at[list(idxs)[0],"nodeID"]
            #nodeID_1 = nodes_gdf.at[list(idxs)[1],"nodeID"]
            nodeID_0,nodeID_1 = nodes_gdf.loc[list(idxs),"nodeID"]
            new_geom = Point(x+dx,y)
            print(f"Overlapping nodes {nodeID_0} and {nodeID_1}.")
            #edges starting in shifted node
            mask_u = edges_gdf["u"] == nodeID_0
            if mask_u.any():
                edges_gdf.loc[mask_u,"geometry"] = edges_gdf.loc[mask_u].apply(lambda row:replace_start_end_LineString(row,new_geom,True),axis=1)
            #edges ending in shifted node
            mask_v = edges_gdf["v"] == nodeID_0
            if mask_v.any():
                edges_gdf.loc[mask_v,"geometry"] = edges_gdf.loc[mask_v].apply(lambda row:replace_start_end_LineString(row,new_geom,False),axis=1)
            nodes_gdf.at[list(idxs)[0],"geometry"] = new_geom
        elif len(idxs) > 2:
            print("More than 2 overlapping nodes.")
    nodes_gdf = nodes_gdf.drop(columns=["x","y"])
    return nodes_gdf,edges_gdf

def downloadStreetGraph(location_i,boundary_gdf):
    if location_i == 0:
        G = ox.graph_from_place(places[location_i], network_type=network_type[location_i],retain_all=True)
    elif location_i == 1:
        polygon = boundary_gdf.to_crs("EPSG:4326").geometry.iloc[0]
        G = ox.graph_from_polygon(polygon, network_type=network_type[location_i],retain_all=True)
    print("Download of street graph finished.")
    #convert to graph
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    gdf_nodes = gdf_nodes.reset_index()
    gdf_nodes = gdf_nodes.rename(columns={'osmid':'nodeID','street_count':'s_count'})
    gdf_nodes["nodeID"] = gdf_nodes["nodeID"].astype("int64") 
    gdf_nodes.drop(['y','x','highway','railway'],axis=1,errors='ignore',inplace=True)
    # We need an unique ID for each edge, nodes already have
    # osmid appears multiple times
    # we have one edge per direction, so two edges over each other
    gdf_edges["edgeID"] = np.arange(0, gdf_edges.shape[0], dtype='int')
    gdf_edges = gdf_edges.reset_index()
    gdf_edges.drop(['key','name','oneway','reversed','service','access','ref','bridge','width','tunnel','junction','area'],axis=1,errors='ignore',inplace=True)

    gdf_edges_utm = gdf_edges.to_crs(utm_crs[location_i])
    gdf_nodes_utm = gdf_nodes.to_crs(utm_crs[location_i])


    highway_feature_gdf = get_highway_feature_gdf(location_i,boundary_gdf)
    print("Download of Highway Features finished.")

    #fix listified columns
    listified_columns = ["highway","maxspeed","lanes"]
    gdf_edges_utm = fix_listed_columns(gdf_edges_utm,listified_columns,location_i,highway_feature_gdf)
    gdf_edges_utm["osmid"] = gdf_edges_utm["osmid"].astype("int64")
    gdf_edges_utm[listified_columns] = gdf_edges_utm[listified_columns].astype("string")



    print("1",len(gdf_edges_utm))
    gdf_edges_utm = add_geo_info(gdf_edges_utm,location_i,boundary_gdf,highway_feature_gdf)
    #shift duplicate nodes by small amount (only happens in Beijing very rarely)
    gdf_nodes_utm,gdf_edges_utm = shift_overlapping_nodes(gdf_nodes_utm,gdf_edges_utm)
    print("end",len(gdf_edges_utm))
    print("edgeID unique",gdf_edges_utm["edgeID"].is_unique)
    # save the nodes and edges
    folder = os.path.join(street_graph,subfolder_name[location_i])
    os.makedirs(folder, exist_ok=True)
    gdf_nodes_utm.to_file(os.path.join(folder,"nodes_"+crs_filename_append[location_i]+".shp"))
    gdf_edges_utm.to_file(os.path.join(folder,"edges_"+crs_filename_append[location_i]+".shp"))


#########################################################################################################

def line_to_points(trip):
    interpolate_dist = 1
    if trip.geom_type=="LineString":
        num_points= int(trip.length // interpolate_dist)+1
        distances = np.linspace(0,trip.length,num_points)
        return [trip.interpolate(d) for d in distances]
    #obsolte now but whatever
    elif trip.geom_type=="MultiLineString":
        point_list = []
        for part in trip.geoms:
             num_points= int(part.length // interpolate_dist)+1
             distances = np.linspace(0,part.length,num_points)
             point_list.extend([part.interpolate(d) for d in distances])
        return point_list
    else:
        raise ValueError("Unexpected geometry type",trip.geom_type)

def to_file(row,location_i,folder):
    points = row["points"]
    tripID = int(row["TripID"])
    gdf = gpd.GeoDataFrame(geometry=points,crs=utm_crs[location_i])
    gdf["fid"] = gdf.index
    name = "tra_"+str(tripID)+".shp"
    target= os.path.join(folder,name)
    gdf.to_file(target)

def split_multilines(trip_gdf,location_i):
    trip_gdf["TripID"] = trip_gdf["TripID"].astype(int)
    multis = trip_gdf[trip_gdf.geom_type == "MultiLineString"]
    exploded = multis.explode()
    no_multies = trip_gdf[trip_gdf.geom_type != "MultiLineString"]
    max_trip_id = no_multies["TripID"].max()
    exploded["TripID"] = np.arange(max_trip_id+1,exploded.shape[0]+max_trip_id+1,dtype='int')
    result = pd.concat((no_multies,exploded),axis=0)
    #sanity checks
    #check if ID unique
    if not result["TripID"].is_unique:
        raise ValueError("ID is not unique somehow.")
    #only LineString type
    if len(result[result.geom_type =="LineString"]) != len(result):
        raise ValueError("Geometry other than LineString encountered")
    #check if no self intersections: geometry.is_simple
    if len(result[~result.geometry.apply(lambda geom: geom.is_simple)]) > 0:
        raise ValueError("Some LineString has self intersection")
    columns_to_keep = ["TripID","Purpose","Start","Stop","UserId","age","gender","income","ethnicity","cycling_fr","rider_hist","rider_type","geometry"]
    result = result[columns_to_keep]
    return result.to_crs(utm_crs[location_i])

def philadelphia_to_points(location_i,boundary_gdf):
    #in meters
    folder = os.path.join(travel_data,subfolder_name[location_i])
    file = os.path.join(folder,"tripbytrip_rd2.shp")
    trips_gdf = gpd.read_file(file).to_crs(utm_crs[location_i])
    #remove all trips going beyond border of philadephia
    trips_filt = gpd.sjoin(trips_gdf,boundary_gdf,how="inner",predicate="within")
    print("After clipping: "+str(len(trips_filt))+" trips.")
    #split all multilinestrings to single lines and check if there are any self intersections afterwards. this will save us headaches
    trips_cleaned_gdf = split_multilines(trips_filt,location_i)
    clean_travel_folder = os.path.join(cleaned_travel,subfolder_name[location_i])
    os.makedirs(clean_travel_folder, exist_ok=True)
    clean_travel_file = os.path.join(clean_travel_folder,"tripsCleaned_"+crs_filename_append[location_i]+".shp")
    trips_cleaned_gdf.to_file(clean_travel_file)
    trips_cleaned_gdf["points"] = trips_cleaned_gdf["geometry"].apply(line_to_points) 
    dest_folder = os.path.join(dest,subfolder_name[location_i])
    os.makedirs(dest_folder, exist_ok=True)
    trips_cleaned_gdf.apply(lambda row: to_file(row,location_i,dest_folder) ,axis=1) 

##########################################

def plt_to_gdf(subdir,file,location_i):
    p = os.path.join(subdir,file)
    with open(p, "r") as f:
        lines = f.readlines()
    #strip first 6
    lines = lines[6:]
    split_lines = [line.split(",") for line in lines]
    cleaned_lines = [line[:2]+[line[5]]+[line[6][:-1]] for line in split_lines  ]
    gdf = gpd.GeoDataFrame(cleaned_lines,columns=["lat","lon","date","time"])
    gdf["geo"] = gdf.apply(lambda row: Point(row["lon"],row["lat"]),axis=1)
    gdf = gdf.rename(columns={"geo":"geometry"})
    gdf = gdf.set_geometry('geometry')
    gdf.set_crs(epsg=4326, inplace=True)
    gdf = gdf.to_crs(utm_crs[location_i])
    gdf["fid"] = range(len(gdf))
    return gdf

def only_inside(gdf,boundary_gdf):
    gdf_within = gpd.sjoin(gdf,boundary_gdf,how="inner",predicate="within")
    return gdf_within[["fid","lat","lon","date","time","geometry"]]

def add_time_cast_to_china(gdf):
    gdf["date_time"] = gdf["date"] + " " +gdf["time"]
    gdf["date_time"] = pd.to_datetime(gdf['date_time'],format='ISO8601')
    #all times in Geolife are in GMT
    gdf["date_time"] = gdf["date_time"].dt.tz_localize('GMT')
    gdf["date_time"] = gdf["date_time"].dt.tz_convert('PRC')
    gdf["date"] = gdf["date_time"].dt.strftime('%Y-%m-%d')
    gdf['time'] = gdf['date_time'].dt.strftime('%H:%M:%S')
    return gdf


time_interval = pd.Timedelta(minutes=1)
def trajectory_segmentation(gdf):
    gdf['time_diff'] = gdf["date_time"] - gdf['date_time'].shift(1)  
    gdf["gap"] = gdf['time_diff'] > time_interval
    gdf["gap"] = gdf["gap"].fillna(False)
    gdf['group'] = gdf['gap'].cumsum().astype(int)
    grouped_dfs = [group[["fid","lat","lon","date","time","date_time","geometry"]]  for _, group in gdf.groupby('group')]
    return grouped_dfs

#lables.txt file to dataframe
def get_labels(label_path):
    with open(label_path, "r") as f:
        lines = f.readlines()
    lines = lines[1:]
    split_lines = [line.rstrip("\n").split("\t") for line in lines]
    df = pd.DataFrame(split_lines,columns=["start_date_time","end_date_time","transportation"])
    df["start_date_time"] = pd.to_datetime(df['start_date_time'],format='ISO8601')
    df["end_date_time"] = pd.to_datetime(df['end_date_time'],format='ISO8601')
    df["start_date_time"] = df["start_date_time"].dt.tz_localize('GMT')
    df["end_date_time"] = df["end_date_time"].dt.tz_localize('GMT')
    df["start_date_time"] = df["start_date_time"].dt.tz_convert('PRC')
    df["end_date_time"] = df["end_date_time"].dt.tz_convert('PRC')
    return df

def remove_overlaps(label_df,UserID):

    result = []
    for index,row in label_df.iterrows():
        if UserID==53:
            print(row)
        s,e,l = row["start_date_time"],row["end_date_time"],row["transportation"]
        new_result = []
        for r in result:
            if UserID==53:
                print(r)
            rs,re,rl = r["start_date_time"],r["end_date_time"],r["transportation"]
            #no overlap, interval in results doesn't overlay current
            if re<=s or rs>=e:
                new_result.append(r)
            #overlap: cut interval in results
            else:
                if rs < s:
                    new_result.append({"start_date_time":rs,"end_date_time":s,"transportation":rl})
                if re > e:
                    new_result.append({"start_date_time":e,"end_date_time":re,"transportation":rl})
        new_result.append({"start_date_time":s,"end_date_time":e,"transportation":l})
        result = sorted(new_result,key=lambda x:x["start_date_time"])
    return pd.DataFrame(result)

#we try to clean time intervals such that in the end
#t1<=t2 car
#<=t3<=t4 walk
#<=t5<=t6
#etc...
def clean_time_intervals(label_df,userID):
    #sanitiy checks
    #this doesn't happen in our data
    if not (label_df["start_date_time"]<=label_df["end_date_time"]).all():
        print("Time interval broken")
        print(label_df)
        return (False,None)
    #this also doesn't happen in our data, is sorted
    if not label_df["start_date_time"].is_monotonic_increasing:
        print("Label start time column not ordered, soring")
        print(label_df)
        label_df = label_df.sort_values(by='start_date_time').reset_index(drop=True)
    #this does happen, some of the fils have overlapping intervals
    if not label_df["end_date_time"].is_monotonic_increasing:
        print("Label end time column not ordered")
        label_df = remove_overlaps(label_df,userID)
    ##see where the start of next label is not larger than end of previous label
    end = label_df["end_date_time"].iloc[:-1]
    start = label_df["start_date_time"].shift(-1).iloc[:-1]
    correct_intervals = start>=end
    #add a True at the beginning. the first start time has no previous end time
    correct_intervals = pd.concat([correct_intervals,pd.Series([True])],ignore_index=True)
    #for the not correct intervals, the start time must be between start and end time of the row above it (see our sanity checks)
    #so we set the end time to the row above to the new start time (assume newer label is more accurate). Now either this interval is 0 seconds long or start where the previous ended
    label_df.loc[(~correct_intervals),"end_date_time"] = label_df["start_date_time"].shift(-1)
    #we remove the zero lenght intervalls
    label_df = label_df[label_df["start_date_time"]!=label_df["end_date_time"] ]
    #check if everything was successful
    end = label_df["end_date_time"].iloc[:-1]
    start = label_df["start_date_time"].shift(-1).iloc[:-1]
    if not (end<=start).all():
        print("Didn't work:(")
        print(label_df)
        return (False,None)
    else:
        return (True,label_df)

def add_label_to_gps_readings(gdf,label_df):
    label_df_indexer = 0
    gdf["label"] = None
    for index, row in gdf.iterrows():
        time = row["date_time"]
        while label_df_indexer<len(label_df) and time > label_df["end_date_time"].iloc[label_df_indexer]:
            label_df_indexer += 1
        if label_df_indexer<len(label_df) and label_df["start_date_time"].iloc[label_df_indexer] <= time <= label_df["end_date_time"].iloc[label_df_indexer]:
            gdf.loc[index,"label"] =  label_df["transportation"].iloc[label_df_indexer]
    return gdf



def beijing_to_points(location_i,boundary_gdf):
    tripID = 0
    trip_dic = {"TripID":[],"UserId":[],"label":[]}     
    dest_folder = os.path.join(dest,subfolder_name[location_i])
    os.makedirs(dest_folder, exist_ok=True)
    rootdir = os.path.join(travel_data,subfolder_name[location_i])
    bad_labels = 0
    for subdir, dirs, files in os.walk(rootdir):
        head,tail = os.path.split(subdir)
        if tail == "Trajectory":
            _,userID = os.path.split(head)
            userID = int(userID)
            label_path = os.path.join(head,"labels.txt")
            label_exist = os.path.exists(label_path)
            if label_exist:
                label_df = get_labels(label_path)
                #sometimes label time intervals broken
                is_fixed,label_df = clean_time_intervals(label_df,userID)
                if not(is_fixed):
                    label_exist = False
                    print("Discarding non valid label. User: ",userID)
                    bad_labels += 1
            for file in files:
                gdf = plt_to_gdf(subdir,file,location_i)
                gdf_filt = only_inside(gdf,boundary_gdf)
                if len(gdf_filt)==0:
                    continue
                #just to make sure the join doesn't break the order
                gdf_filt = gdf_filt.sort_values(by='fid').reset_index(drop=True)
                gdf_time = add_time_cast_to_china(gdf_filt)
                list_of_gdf_segments = trajectory_segmentation(gdf_time) 
                for sub_gdf in list_of_gdf_segments:
                    sub_gdf = sub_gdf.reset_index(drop=True)
                    if label_exist:
                        sub_gdf = add_label_to_gps_readings(sub_gdf,label_df)
                    else:
                        sub_gdf["label"] = None
                    change_points = sub_gdf['label'].ne(sub_gdf['label'].shift(1)).cumsum()
                    sub_sub_gdfs = [group for _, group in sub_gdf.groupby(change_points)]
                    for sub_sub_gdf in sub_sub_gdfs:
                        trip_dic["TripID"].append(tripID)
                        trip_dic["UserId"].append(userID)
                        trip_dic["label"].append(sub_sub_gdf["label"].iloc[0])
                        point_trace_loc = os.path.join(dest_folder,"tra_"+str(tripID)+".shp")
                        sub_sub_gdf[["fid","lat","lon","date","time","geometry"]].to_file(point_trace_loc)
                        if tripID%100==0:
                            print(tripID)
                        tripID += 1
    trip_dest_folder = os.path.join(cleaned_travel,subfolder_name[location_i])
    os.makedirs(trip_dest_folder,exist_ok=True)
    trip_file = os.path.join(trip_dest_folder,"trips.csv")
    tri_df = pd.DataFrame.from_dict(trip_dic)
    tri_df.to_csv(trip_file)
    print("Bad labels: ",bad_labels)

data_to_point_trajectories = [philadelphia_to_points,beijing_to_points]

location_i = 1
def main():
    print(places[location_i])
    boundary_gdf = get_boundary(location_i)
    downloadStreetGraph(location_i,boundary_gdf)
    data_to_point_trajectories[location_i](location_i,boundary_gdf)
    


if __name__ == "__main__":
    main()
