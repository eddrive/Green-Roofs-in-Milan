# Import necessary libraries
import math  # For mathematical calculations such as trigonometric and distance functions
import json  # For reading and parsing JSON files
import geopandas as gpd  # For working with geospatial data (GeoJSON, shapefiles, etc.)
import folium  # For creating interactive maps in HTML format

# Function to calculate the distance between two geographic points using the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the distance in kilometers between two geographic coordinates.
    Uses the Haversine formula, which accounts for the curvature of the Earth.
    
    Parameters:
    lat1, lon1: Latitude and longitude of the first point
    lat2, lon2: Latitude and longitude of the second point
    
    Returns:
    Distance in kilometers between the two points
    """
    R = 6371  # Earth's radius in kilometers
    # Convert latitude and longitude differences to radians
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    # Haversine formula to calculate the great-circle distance
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    
    # Convert the central angle to distance
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Function to calculate a weighted mean of environmental indices (temperature, precipitation, pollution)
def calculate_weighted_mean(element, index_type, locations, station_values):
    """
    Computes the weighted mean of index values for a given element and index type.
    The weighted mean is based on the inverse of the distance to the two nearest stations.
    
    Parameters:
    element: A geometry object (e.g., rooftop polygon)
    index_type: The type of index ('temperature', 'precipitation', 'pollution')
    locations: Dictionary of station coordinates
    station_values: Dictionary of environmental index values for stations
    
    Returns:
    Weighted mean value for the specified index type
    """
    if element is None or element.is_empty:
        return 0  # Return 0 if the geometry is invalid or empty
    
    # Calculate the centroid of the rooftop geometry
    centroid = element.centroid
    element_coords = (centroid.y, centroid.x)
    
    # Get the relevant stations for the specified index type
    stations = locations.get(index_type, {})
    
    # Calculate distances to all relevant stations using the Haversine formula
    distances = {
        station: haversine(element_coords[0], element_coords[1], coords[0], coords[1])
        for station, coords in stations.items()
    }
    
    # Find the two closest stations
    closest_stations = sorted(distances.items(), key=lambda x: x[1])[:2]
    
    # Calculate weights inversely proportional to the distance
    weights = {}
    for station, distance in closest_stations:
        weights[station] = 1 / distance if distance != 0 else 1e6  # Avoid division by zero
    
    # Normalize the weights so that they sum up to 1
    total_weight = sum(weights.values())
    normalized_weights = {station: weight / total_weight for station, weight in weights.items()}
    
    # Calculate the weighted mean of the environmental index values
    weighted_sum = sum(
        station_values.get(index_type, {}).get(station, 0) * normalized_weights[station]
        for station in normalized_weights
    )
    
    return weighted_sum


# Function to calculate weighted indices for each rooftop in the GeoJSON file
def calculate_weighted_indices(geojson, input_json):
    """
    Computes weighted indices for temperature, pollution, and precipitation for each rooftop.
    
    Parameters:
    geojson: Path to the GeoJSON file containing rooftop data
    input_json: Path to the input JSON file with station data and environmental values
    
    Returns:
    GeoDataFrame with new columns for each computed index
    """
    # Load the rooftops data as a GeoDataFrame
    gdf = gpd.read_file(geojson)
    
    # Drop unnecessary columns if present
    gdf = gdf.drop(columns=["IMPACT", "NOTE", "legenda1"], errors='ignore')
    
    # Load station data and environmental index values from the JSON file
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # Prepare dictionaries for station coordinates and index values
    locations = {key: {k: v['coordinates'] for k, v in data['stations'][key].items()} for key in ['temperature', 'precipitation', 'pollution']}
    station_values = {key: {k: v['value'] for k, v in data['stations'][key].items()} for key in ['temperature', 'precipitation', 'pollution']}
    
    # Compute weighted indices for each environmental parameter
    for index_type in ['temperature', 'pollution', 'precipitation']:
        gdf[f"{index_type}_index"] = gdf.geometry.apply(lambda geom: calculate_weighted_mean(geom, index_type, locations, station_values))
    
    return gdf


# Function to compute a general impact index as a weighted sum of normalized indices
def calculate_general_index(gdf, input_json):
    """
    Computes a general index as a weighted sum of normalized indices.
    
    Parameters:
    gdf: GeoDataFrame with individual environmental indices
    input_json: Path to the input JSON file containing weight configuration
    
    Returns:
    GeoDataFrame with an additional column for the general index
    """
    with open(input_json, 'r') as f:
        data = json.load(f)['input']
    
    # Extract weights for each index type
    weights = {k: data.get(k, 0) for k in ['temperature_weight', 'pollution_weight', 'precipitation_weight']}
    
    # Ensure the weights are normalized
    total_weight = sum(weights.values())
    if total_weight == 0:
        raise ValueError("The sum of weights cannot be zero.")
    
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Normalize each environmental index column
    for key in weights:
        index_column = key.replace("_weight", "") + "_index"
        
        # Check if the column exists
        if index_column not in gdf.columns:
            raise KeyError(f"Column '{index_column}' not found in GeoDataFrame. Available columns: {list(gdf.columns)}")
        
        # Normalize values between 0 and 1
        min_val = gdf[index_column].min()
        max_val = gdf[index_column].max()

        gdf[f"{index_column}_normalized"] = (
            (gdf[index_column] - min_val) / (max_val - min_val) if max_val != min_val else 0
        )

    # Calculate the general index as a weighted sum of normalized indices
    gdf["general_index"] = sum(weights[k] * gdf[f"{k.replace('_weight', '')}_index_normalized"] for k in weights)

    return gdf

# Function to calculate the total cost for converting rooftops into green roofs
def calculate_total_cost(gdf, input_json):
    """
    Computes the total cost for each element in the GeoDataFrame.
    
    Parameters:
    gdf: GeoDataFrame containing rooftop data with 'area' field
    input_json: Path to the input JSON file with the cost per square meter
    
    Returns:
    GeoDataFrame with an additional 'total_cost' column
    """
    # Load the cost per square meter from the input JSON file
    with open(input_json, 'r') as f:
        cost_per_square_meter = json.load(f)['input']['cost']
    
    # Ensure the 'area' column exists in the GeoDataFrame
    if 'area' not in gdf.columns:
        raise KeyError("The 'area' field is missing in the GeoJSON.")
    
    # Calculate the total cost as area * cost per square meter
    gdf["total_cost"] = gdf["area"] * cost_per_square_meter
    
    return gdf

# Function to select rooftops that fit within the budget while maximizing the general index
def select_elements_within_budget(gdf, input_json):
    """
    Selects elements within budget, sorted by the general index.
    
    Parameters:
    gdf: GeoDataFrame with 'general_index' and 'total_cost' columns
    input_json: Path to the input JSON file with budget and property filter settings
    
    Returns:
    GeoDataFrame containing the selected rooftops
    """
    # Load budget and filter settings from the input JSON file
    with open(input_json, 'r') as f:
        data = json.load(f)['input']
        budget = data.get('budget', 0) 
        only_comune_milano = data.get('only_comune_milano', False)  # Default to False if not found

    # Filter rooftops by ownership if required
    if only_comune_milano:
        gdf = gdf[gdf["PROPRIETA"] == "COMUNE MILANO"]
    
    # Sort rooftops by the general index in descending order
    gdf_sorted = gdf.sort_values(by="general_index", ascending=False)
    
    selected_elements = []
    total_spent = 0
    
    # Iterate through rooftops and select those within the budget
    for _, row in gdf_sorted.iterrows():
        if total_spent + row["total_cost"] <= budget:
            selected_elements.append(row)
            total_spent += row["total_cost"]
        else:
            break
    
    # Return a new GeoDataFrame with the selected rooftops
    return gpd.GeoDataFrame(selected_elements, crs=gdf.crs)

# Function to visualize selected rooftops on an interactive map
def visualize_on_map(geojson_path, map_center=[45.4642, 9.1900], zoom_start=12, output_path="Selected_Rooftops_Map.html"):
    """
    Visualizes a GeoJSON file on an interactive map using Folium.
    
    Parameters:
    geojson_path: Path to the GeoJSON file to visualize
    map_center: Coordinates [lat, lon] to center the map (default is Milan)
    zoom_start: Initial zoom level for the map
    output_path: The file path to save the HTML map
    """
    # Initialize the Folium map centered on Milan
    m = folium.Map(location=map_center, zoom_start=zoom_start)

    # Define the styling for municipal and private rooftops
    def style_function(feature):
        color = 'red' if feature['properties']['PROPRIETA'] == 'COMUNE MILANO' else 'blue'
        return {
            'fillColor': '#228B22',  # Green fill color for rooftops
            'color': color,           # Red or blue border based on ownership
            'weight': 2,              # Border line thickness
            'fillOpacity': 0.6        # Transparency of the fill color
        }

    # Add the GeoJSON layer to the Folium map
    folium.GeoJson(
        geojson_path,
        name="Selected Green Roofs",
        style_function=style_function
    ).add_to(m)

    # Add a legend to explain the color coding
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

    # Add layer controls to toggle the visibility of map elements
    folium.LayerControl().add_to(m)

    # Save the map as an HTML file and provide user feedback
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