import math
import json
import geopandas as gpd

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the distance in kilometers between two geographic coordinates."""
    R = 6371  # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def calculate_weighted_mean(element, index_type, locations, station_values):
    """
    Calculates the weighted mean of index values for a given element and index type.
    """
    centroid = element.centroid
    element_coords = (centroid.y, centroid.x)
    stations = locations.get(index_type, {})
    
    distances = {
        station: haversine(element_coords[0], element_coords[1], coords[0], coords[1])
        for station, coords in stations.items()
    }
    
    closest_stations = sorted(distances.items(), key=lambda x: x[1])[:2]
    weights = {station: 1 / distance if distance != 0 else 1e6 for station, distance in closest_stations}
    total_weight = sum(weights.values())
    normalized_weights = {station: weight / total_weight for station, weight in weights.items()}
    
    return sum(station_values.get(index_type, {}).get(station, 0) * normalized_weights[station]
               for station in normalized_weights)

def calculate_weighted_indices(geojson, input_json):
    """
    Computes weighted indices for temperature, pollution, and precipitation.
    """
    gdf = gpd.read_file(geojson) if isinstance(geojson, str) else geojson
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    locations = {key: {k: v['coordinates'] for k, v in data['stations'][key].items()} for key in ['temperature', 'precipitation', 'pollution']}
    station_values = {key: {k: v['value'] for k, v in data['stations'][key].items()} for key in ['temperature', 'precipitation', 'pollution']}
    
    for index_type in ['temperature', 'pollution', 'precipitation']:
        gdf[f"{index_type}_index"] = gdf.geometry.apply(lambda geom: calculate_weighted_mean(geom, index_type, locations, station_values))
    
    return gdf

def calculate_general_index(geojson, input_json):
    """
    Computes a general index as a weighted sum of normalized indices.
    """
    gdf = gpd.read_file(geojson) if isinstance(geojson, str) else geojson
    
    with open(input_json, 'r') as f:
        data = json.load(f)['input']
    
    weights = {k: data[k] for k in ['temperature_weight', 'pollution_weight', 'precipitation_weight']}
    total_weight = sum(weights.values())
    
    if total_weight == 0:
        raise ValueError("The sum of weights cannot be zero.")
    
    weights = {k: v / total_weight for k, v in weights.items()}
    
    for key in weights:
        min_val = gdf[f"{key}_index"].min()
        max_val = gdf[f"{key}_index"].max()
        gdf[f"{key}_index_normalized"] = (gdf[f"{key}_index"] - min_val) / (max_val - min_val)
    
    gdf["general_index"] = sum(weights[k] * gdf[f"{k}_index_normalized"] for k in weights)
    
    return gdf

def calculate_total_cost(geojson, cost_per_square_meter):
    """
    Computes the total cost for each element in the GeoDataFrame.
    """
    gdf = gpd.read_file(geojson) if isinstance(geojson, str) else geojson
    
    if 'area' not in gdf.columns:
        raise KeyError("The 'area' field is missing in the GeoJSON.")
    
    gdf["total_cost"] = gdf["area"] * cost_per_square_meter
    
    return gdf

def select_elements_within_budget(geojson, input_json, only_comune_milano=False):
    """
    Selects elements within budget, sorted by the general index.
    """
    gdf = gpd.read_file(geojson) if isinstance(geojson, str) else geojson
    
    with open(input_json, 'r') as f:
        budget = json.load(f)['input']['budget']
    
    if only_comune_milano:
        gdf = gdf[gdf["PROPRIETA"] == "COMUNE MILANO"]
    
    gdf_sorted = gdf.sort_values(by="general_index", ascending=False)
    
    selected_elements = []
    total_spent = 0
    
    for _, row in gdf_sorted.iterrows():
        if total_spent + row["total_cost"] <= budget:
            selected_elements.append(row)
            total_spent += row["total_cost"]
        else:
            break
    
    return gpd.GeoDataFrame(selected_elements, crs=gdf.crs)

# Example usage
updated_gdf = calculate_weighted_indices(gdf, "input.json")
updated_gdf = calculate_general_index(updated_gdf, "input.json")
updated_gdf = calculate_total_cost(updated_gdf, cost_per_square_meter=80)
selected_rooftops = select_elements_within_budget(updated_gdf, "input.json", only_comune_milano=True)
