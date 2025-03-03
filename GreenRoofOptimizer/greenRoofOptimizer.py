import math
import json
import geopandas as gpd
import folium

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
    Computes the weighted mean of index values for a given element and index type.
    """
    if element is None or element.is_empty:
        return 0  # If the geometry is null, return 0
    
    centroid = element.centroid
    element_coords = (centroid.y, centroid.x)
    stations = locations.get(index_type, {})
    
    # Compute distances to stations
    distances = {
        station: haversine(element_coords[0], element_coords[1], coords[0], coords[1])
        for station, coords in stations.items()
    }
    
    # Find the two closest stations
    closest_stations = sorted(distances.items(), key=lambda x: x[1])[:2]
    
    # Compute and normalize weights
    weights = {}
    for station, distance in closest_stations:
        weights[station] = 1 / distance if distance != 0 else 1e6  # Avoid division by zero
    
    total_weight = sum(weights.values())  # Total weight sum
    normalized_weights = {station: weight / total_weight for station, weight in weights.items()}  # Normalize
    
    # Compute weighted mean
    weighted_sum = sum(
        station_values.get(index_type, {}).get(station, 0) * normalized_weights[station]
        for station in normalized_weights
    )
    
    return weighted_sum

def calculate_weighted_indices(geojson, input_json):
    """
    Computes weighted indices for temperature, pollution, and precipitation.
    """
    gdf = gpd.read_file(geojson)
    gdf = gdf.drop(columns=["IMPACT", "NOTE", "legenda1"], errors='ignore')
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    locations = {key: {k: v['coordinates'] for k, v in data['stations'][key].items()} for key in ['temperature', 'precipitation', 'pollution']}
    station_values = {key: {k: v['value'] for k, v in data['stations'][key].items()} for key in ['temperature', 'precipitation', 'pollution']}
    
    for index_type in ['temperature', 'pollution', 'precipitation']:
        gdf[f"{index_type}_index"] = gdf.geometry.apply(lambda geom: calculate_weighted_mean(geom, index_type, locations, station_values))
    
    return gdf

def calculate_general_index(gdf, input_json):
    """
    Computes a general index as a weighted sum of normalized indices.
    """
    with open(input_json, 'r') as f:
        data = json.load(f)['input']
    
    # Ensure weights are not None
    weights = {k: data.get(k, 0) for k in ['temperature_weight', 'pollution_weight', 'precipitation_weight']}
    
    total_weight = sum(weights.values())

    if total_weight == 0:
        raise ValueError("The sum of weights cannot be zero.")

    # Normalize weights
    weights = {k: v / total_weight for k, v in weights.items()}

    # Normalize each index
    for key in weights:
        index_column = key.replace("_weight", "") + "_index"  # Correct column name
        
        if index_column not in gdf.columns:
            raise KeyError(f"Column '{index_column}' not found in GeoDataFrame. Available columns: {list(gdf.columns)}")

        min_val = gdf[index_column].min()
        max_val = gdf[index_column].max()

        gdf[f"{index_column}_normalized"] = (
            (gdf[index_column] - min_val) / (max_val - min_val) if max_val != min_val else 0
        )

    # Compute general index as weighted sum
    gdf["general_index"] = sum(weights[k] * gdf[f"{k.replace('_weight', '')}_index_normalized"] for k in weights)

    return gdf
 

def calculate_total_cost(gdf, input_json):
    """
    Computes the total cost for each element in the GeoDataFrame.
    """
    
    with open(input_json, 'r') as f:
        cost_per_square_meter = json.load(f)['input']['cost']
    
    if 'area' not in gdf.columns:
        raise KeyError("The 'area' field is missing in the GeoJSON.")
    
    gdf["total_cost"] = gdf["area"] * cost_per_square_meter
    
    return gdf

def select_elements_within_budget(gdf, input_json):
    """
    Selects elements within budget, sorted by the general index.
    """
    
    with open(input_json, 'r') as f:
        data = json.load(f)['input']
        budget = data.get('budget', 0) 
        only_comune_milano = data.get('only_comune_milano', False)  # Default to False if not found

    
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

def visualize_on_map(geojson_path, map_center=[45.4642, 9.1900], zoom_start=12, output_path="Selected_Rooftops_Map.html"):
    """
    Visualizes a GeoJSON file on an interactive map using Folium.
    
    Args:
        geojson_path (str): Path to the GeoJSON file to visualize.
        map_center (list): Coordinates [lat, lon] to center the map.
        zoom_start (int): Initial zoom level for the map.
        output_path (str): The file path to save the HTML map.
    """
    # Create a Folium map centered on Milan
    m = folium.Map(location=map_center, zoom_start=zoom_start)

    def style_function(feature):
        color = 'red' if feature['properties']['PROPRIETA'] == 'COMUNE MILANO' else 'blue'
        return {
            'fillColor': '#228B22',  # Fill color (green)
            'color': color,           # Border color
            'weight': 2,              # Border thickness
            'fillOpacity': 0.6
        }

    folium.GeoJson(
        geojson_path,
        name="Selected Green Roofs",
        style_function=style_function
    ).add_to(m)

    # Add legend to the map
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 100px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white; opacity: 0.8;
                padding: 10px;">
    <b>Property:</b><br>
    <i style="background: red; width: 20px; height: 20px; display: inline-block;"></i> Municipal<br>
    <i style="background: blue; width: 20px; height: 20px; display: inline-block;"></i> Private
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)

    # Save and display the map
    m.save(output_path)
    print(f"Map saved as '{output_path}'")



# Example usage
gdf = "ds1446_tetti_verdi_potenziali.geojson"
updated_gdf = calculate_weighted_indices(gdf, "input.json")
updated_gdf = calculate_general_index(updated_gdf, "input.json")
updated_gdf = calculate_total_cost(updated_gdf, "input.json")
selected_rooftops = select_elements_within_budget(updated_gdf, "input.json")

with open("input.json", 'r') as f:
    data = json.load(f)['input']

budget = data['budget']
temperature_weight = data['temperature_weight']
precipitation_weight = data['precipitation_weight']
pollution_weight = data['pollution_weight']
municipal_only = data['only_comune_milano']

municipal_tag = 'Municipal' if municipal_only else 'All'

output_name = f"SelectedGR_{budget}_Temp{temperature_weight}_Prec{precipitation_weight}_Pol{pollution_weight}_{municipal_tag}"

# Generate filenames
geojson_output = f"{output_name}.geojson"
html_output = f"{output_name}.html"

# Save the selected rooftops to a new GeoJSON file
selected_rooftops.to_file(geojson_output, driver="GeoJSON")

# Visualize the result on an interactive map
visualize_on_map(geojson_path=geojson_output, output_path=html_output)

