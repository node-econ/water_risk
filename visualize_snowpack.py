#!/usr/bin/env python3
"""
Snowpack Visualization

Creates an interactive map showing HUC-12 watersheds with SNOTEL station
locations colored by their Basin Index (% of median snowpack).

Usage:
    python visualize_snowpack.py --state OR
    python visualize_snowpack.py --state CA --output ca_snowpack.html
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
from branca.colormap import LinearColormap
from folium.plugins import MarkerCluster

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, STATES_OF_INTEREST
from nwcc_api import NWCCClient


def get_snotel_data(state: str, date: str = None) -> gpd.GeoDataFrame:
    """
    Fetch SNOTEL station data and return as GeoDataFrame.
    
    Args:
        state: Two-letter state code
        date: Date string (YYYY-MM-DD), defaults to yesterday
        
    Returns:
        GeoDataFrame with station locations and snowpack data
    """
    if date is None:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    client = NWCCClient()
    
    # Get stations
    print(f"Fetching {state} SNOTEL stations...")
    stations = client.get_stations(state=state, network="SNTL")
    stations_dict = {s["stationTriplet"]: s for s in stations}
    triplets = list(stations_dict.keys())
    
    print(f"Found {len(triplets)} active stations")
    
    # Get data
    print("Fetching snow data...")
    data = client.get_data(
        station_triplets=triplets,
        elements=["WTEQ", "SNWD"],
        begin_date=date,
        end_date=date,
        include_median=True,
        batch_size=10,
    )
    
    # Process results
    results = []
    for station_data in data:
        triplet = station_data.get("stationTriplet", "")
        station_info = stations_dict.get(triplet, {})
        
        lat = station_info.get("latitude")
        lon = station_info.get("longitude")
        
        if lat is None or lon is None:
            continue
        
        wteq = wteq_med = snwd = None
        
        for elem_data in station_data.get("data", []):
            elem = elem_data.get("stationElement", {}).get("elementCode", "")
            values = elem_data.get("values", [])
            
            if values:
                latest = values[-1]
                val = latest.get("value")
                
                if val is not None and val != -9999.9:
                    if elem == "WTEQ":
                        wteq = val
                        wteq_med = latest.get("median")
                    elif elem == "SNWD":
                        snwd = val
        
        # Calculate basin index (% of median)
        basin_index = None
        if wteq is not None and wteq_med is not None and wteq_med > 0:
            basin_index = (wteq / wteq_med) * 100
        elif wteq == 0:
            basin_index = 0
            
        results.append({
            "triplet": triplet,
            "name": station_info.get("name", triplet),
            "latitude": lat,
            "longitude": lon,
            "elevation": station_info.get("elevation", 0),
            "huc": station_info.get("huc", ""),
            "wteq": wteq,
            "median": wteq_med,
            "basin_index": basin_index,
            "snow_depth": snwd,
        })
    
    # Create GeoDataFrame
    from shapely.geometry import Point
    
    geometry = [Point(r["longitude"], r["latitude"]) for r in results]
    gdf = gpd.GeoDataFrame(results, geometry=geometry, crs="EPSG:4326")
    
    print(f"Retrieved data for {len(gdf)} stations")
    return gdf


def load_huc12_watersheds(
    states: list[str] = None,
    bbox: tuple = None,
    max_watersheds: int = 2000,
) -> gpd.GeoDataFrame:
    """
    Load HUC-12 watersheds, optionally filtered to specific states or bounding box.
    
    Args:
        states: List of state codes to filter to (e.g., ['OR', 'CA'])
        bbox: Optional (minx, miny, maxx, maxy) to spatially filter
        max_watersheds: Maximum number of watersheds to return
        
    Returns:
        GeoDataFrame of HUC-12 watersheds
    """
    wbd_path = RAW_DATA_DIR / "watershed/national/WBD_National_GDB/WBD_National_GDB.gdb"
    
    if not wbd_path.exists():
        raise FileNotFoundError(f"WBD geodatabase not found at {wbd_path}")
    
    print("Loading HUC-12 watersheds (this may take a moment)...")
    
    # Use bbox to limit initial load if provided
    if bbox:
        watersheds = gpd.read_file(wbd_path, layer="WBDHU12", bbox=bbox)
        print(f"Loaded {len(watersheds)} watersheds in bounding box")
    else:
        watersheds = gpd.read_file(wbd_path, layer="WBDHU12")
    
    if states:
        # Filter by state codes in HUC
        # HUC-12 codes starting with specific 2-digit regions for western states
        # This is approximate - HUCs don't align perfectly with state boundaries
        state_huc_prefixes = {
            "CA": ["18"],  # California region
            "NV": ["16"],  # Great Basin 
            "OR": ["17"],  # Pacific Northwest
            "WA": ["17"],  # Pacific Northwest
        }
        
        prefixes = []
        for state in states:
            prefixes.extend(state_huc_prefixes.get(state.upper(), []))
        
        if prefixes:
            prefixes = list(set(prefixes))  # Remove duplicates
            mask = watersheds["huc12"].str[:2].isin(prefixes)
            watersheds = watersheds[mask]
            print(f"Filtered to {len(watersheds)} watersheds in regions: {prefixes}")
    
    # Limit the number of watersheds if too many
    if len(watersheds) > max_watersheds:
        print(f"Limiting to {max_watersheds} watersheds for performance")
        watersheds = watersheds.head(max_watersheds)
    
    return watersheds


def load_state_boundaries(states: list[str]) -> gpd.GeoDataFrame:
    """Load state boundaries for the specified states."""
    states_path = RAW_DATA_DIR / "political/national/tl_2023_us_state"
    shp_file = list(states_path.glob("*.shp"))[0]
    
    all_states = gpd.read_file(shp_file)
    fips_codes = [STATES_OF_INTEREST[s]["fips"] for s in states if s in STATES_OF_INTEREST]
    
    return all_states[all_states["STATEFP"].isin(fips_codes)]


def create_snowpack_map(
    snotel_gdf: gpd.GeoDataFrame,
    watersheds_gdf: gpd.GeoDataFrame = None,
    state_boundaries: gpd.GeoDataFrame = None,
    title: str = "SNOTEL Basin Index Map",
    output_path: str = None,
) -> folium.Map:
    """
    Create an interactive map showing snowpack conditions.
    
    Args:
        snotel_gdf: GeoDataFrame with SNOTEL station data
        watersheds_gdf: Optional GeoDataFrame with HUC-12 boundaries
        state_boundaries: Optional GeoDataFrame with state boundaries
        title: Map title
        output_path: Path to save HTML file
        
    Returns:
        folium.Map object
    """
    # Calculate map center
    center_lat = snotel_gdf.geometry.y.mean()
    center_lon = snotel_gdf.geometry.x.mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles="CartoDB positron",
    )
    
    # Add state boundaries
    if state_boundaries is not None:
        folium.GeoJson(
            state_boundaries.to_crs("EPSG:4326"),
            name="State Boundaries",
            style_function=lambda x: {
                "fillColor": "transparent",
                "color": "#333333",
                "weight": 2,
                "fillOpacity": 0,
            },
        ).add_to(m)
    
    # Add HUC-12 watersheds
    if watersheds_gdf is not None:
        # Simplify for performance and select only needed columns
        watersheds_simple = watersheds_gdf[["name", "huc12", "geometry"]].copy()
        watersheds_simple["geometry"] = watersheds_simple.geometry.simplify(0.01)
        watersheds_simple = watersheds_simple.to_crs("EPSG:4326")
        
        folium.GeoJson(
            watersheds_simple.__geo_interface__,
            name="HUC-12 Watersheds",
            style_function=lambda x: {
                "fillColor": "#e8f4f8",
                "color": "#6baed6",
                "weight": 0.5,
                "fillOpacity": 0.3,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["name", "huc12"],
                aliases=["Watershed:", "HUC-12:"],
            ),
        ).add_to(m)
    
    # Create color map (red to yellow to green, 0-100 scale)
    colormap = LinearColormap(
        colors=["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"],
        vmin=0,
        vmax=100,
        caption="Basin Index (% of Median)"
    )
    colormap.add_to(m)
    
    # Add SNOTEL stations as colored markers
    for idx, row in snotel_gdf.iterrows():
        basin_index = row["basin_index"]
        
        # Determine color
        if basin_index is None:
            color = "#808080"  # Gray for no data
            basin_str = "N/A"
        else:
            # Clamp to 0-150 for color mapping (allow >100%)
            clamped = max(0, min(150, basin_index))
            color = colormap(min(clamped, 100))
            basin_str = f"{basin_index:.0f}%"
        
        # Create popup content
        wteq_str = f"{row['wteq']:.1f}\"" if row['wteq'] is not None else "N/A"
        median_str = f"{row['median']:.1f}\"" if row['median'] is not None else "N/A"
        snwd_str = f"{row['snow_depth']:.0f}\"" if row['snow_depth'] is not None else "N/A"
        
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 200px;">
            <h4 style="margin: 0 0 8px 0; color: #333;">{row['name']}</h4>
            <table style="width: 100%; font-size: 12px;">
                <tr><td><b>Basin Index:</b></td><td style="color: {color}; font-weight: bold;">{basin_str}</td></tr>
                <tr><td><b>WTEQ:</b></td><td>{wteq_str}</td></tr>
                <tr><td><b>Median:</b></td><td>{median_str}</td></tr>
                <tr><td><b>Snow Depth:</b></td><td>{snwd_str}</td></tr>
                <tr><td><b>Elevation:</b></td><td>{row['elevation']:.0f} ft</td></tr>
                <tr><td><b>HUC:</b></td><td>{row['huc']}</td></tr>
            </table>
        </div>
        """
        
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=10,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['name']}: {basin_str}",
            color="#333333",
            weight=1,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
        ).add_to(m)
    
    # Add title
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px 20px; border-radius: 5px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial, sans-serif;">
        <h3 style="margin: 0; color: #333;">{title}</h3>
        <p style="margin: 5px 0 0 0; font-size: 12px; color: #666;">
            Data Date: {(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}
            | Stations: {len(snotel_gdf)}
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save if output path provided
    if output_path:
        m.save(output_path)
        print(f"Map saved to: {output_path}")
    
    return m


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create snowpack visualization map"
    )
    parser.add_argument(
        "--state",
        type=str,
        default="OR",
        help="State code (default: OR)"
    )
    parser.add_argument(
        "--no-watersheds",
        action="store_true",
        default=False,
        help="Skip loading HUC-12 watershed boundaries (faster)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output HTML file path"
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        args.output = str(PROCESSED_DATA_DIR / f"snowpack_map_{args.state.lower()}.html")
    
    # Get SNOTEL data
    snotel_gdf = get_snotel_data(args.state)
    
    # Load watersheds (unless --no-watersheds is specified)
    watersheds_gdf = None
    if not args.no_watersheds:
        try:
            # Calculate bounding box from SNOTEL stations with buffer
            bounds = snotel_gdf.total_bounds  # minx, miny, maxx, maxy
            buffer = 0.5  # degrees
            bbox = (
                bounds[0] - buffer,
                bounds[1] - buffer,
                bounds[2] + buffer,
                bounds[3] + buffer,
            )
            watersheds_gdf = load_huc12_watersheds(states=[args.state], bbox=bbox)
        except FileNotFoundError as e:
            print(f"Warning: Could not load watersheds: {e}")
    
    # Load state boundaries
    try:
        state_boundaries = load_state_boundaries([args.state])
    except Exception as e:
        print(f"Warning: Could not load state boundaries: {e}")
        state_boundaries = None
    
    # Create map
    title = f"{args.state} SNOTEL Basin Index"
    create_snowpack_map(
        snotel_gdf=snotel_gdf,
        watersheds_gdf=watersheds_gdf,
        state_boundaries=state_boundaries,
        title=title,
        output_path=args.output,
    )
    
    # Print summary
    valid_indices = snotel_gdf["basin_index"].dropna()
    if len(valid_indices) > 0:
        print(f"\nBasin Index Summary:")
        print(f"  Stations with data: {len(valid_indices)}")
        print(f"  Average: {valid_indices.mean():.0f}%")
        print(f"  Min: {valid_indices.min():.0f}%")
        print(f"  Max: {valid_indices.max():.0f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

