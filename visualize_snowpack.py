#!/usr/bin/env python3
"""
Snowpack Visualization

Creates an interactive map showing HUC-12 watersheds with SNOTEL station
locations colored by their Basin Index (% of median snowpack).

Usage:
    # Use latest data from database
    python visualize_snowpack.py --state OR
    
    # Specify a date
    python visualize_snowpack.py --state OR --date 2025-01-15
    
    # All states
    python visualize_snowpack.py --all-states --date 2025-12-01
    
    # List available dates
    python visualize_snowpack.py --list-dates
    
    # Use API instead of database (for dates not in DB)
    python visualize_snowpack.py --state OR --use-api
"""

import argparse
import sys
from datetime import datetime, timedelta, date
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from branca.colormap import LinearColormap
from shapely.geometry import Point
from sqlalchemy import text

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, STATES_OF_INTEREST, DATABASE_URL


def get_available_dates():
    """Get list of available dates in the database."""
    from models import get_session
    session = get_session()
    
    result = session.execute(text("""
        SELECT 
            MIN(observation_date) as earliest,
            MAX(observation_date) as latest,
            COUNT(DISTINCT observation_date) as total_dates
        FROM daily_observations
    """)).fetchone()
    
    session.close()
    return {
        "earliest": result[0],
        "latest": result[1],
        "total_dates": result[2],
    }


def get_snotel_data_from_db(
    target_date: date,
    states: list[str] = None,
) -> gpd.GeoDataFrame:
    """
    Fetch SNOTEL data from the PostgreSQL database.
    
    Args:
        target_date: Date to fetch data for
        states: List of state codes (default: all)
        
    Returns:
        GeoDataFrame with station locations and snowpack data
    """
    from models import get_session
    
    if states is None:
        states = list(STATES_OF_INTEREST.keys())
    
    session = get_session()
    
    # Query the database
    query = text("""
        SELECT 
            s.triplet,
            s.name,
            s.state_code,
            s.county_name,
            s.huc,
            s.elevation,
            s.latitude,
            s.longitude,
            d.observation_date,
            d.wteq_value,
            d.wteq_median,
            d.wteq_pct_median as basin_index,
            d.snwd_value as snow_depth,
            d.prec_value as precip,
            d.tmax_value as temp_max,
            d.tmin_value as temp_min
        FROM stations s
        JOIN daily_observations d ON s.id = d.station_id
        WHERE d.observation_date = :target_date
          AND s.state_code = ANY(:states)
          AND s.latitude IS NOT NULL
          AND s.longitude IS NOT NULL
    """)
    
    result = session.execute(query, {
        "target_date": target_date,
        "states": states,
    }).fetchall()
    
    session.close()
    
    if not result:
        print(f"No data found for {target_date}")
        return gpd.GeoDataFrame()
    
    # Convert to DataFrame
    columns = [
        "triplet", "name", "state_code", "county_name", "huc", "elevation",
        "latitude", "longitude", "observation_date", "wteq_value", "wteq_median",
        "basin_index", "snow_depth", "precip", "temp_max", "temp_min"
    ]
    df = pd.DataFrame(result, columns=columns)
    
    # Create geometry
    geometry = [Point(row.longitude, row.latitude) for row in df.itertuples()]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    print(f"Retrieved {len(gdf)} stations for {target_date}")
    return gdf


def get_snotel_data_from_api(
    target_date: date,
    states: list[str] = None,
) -> gpd.GeoDataFrame:
    """
    Fetch SNOTEL data from the NWCC API.
    
    Args:
        target_date: Date to fetch data for
        states: List of state codes (default: all)
        
    Returns:
        GeoDataFrame with station locations and snowpack data
    """
    from nwcc_api import NWCCClient
    
    if states is None:
        states = list(STATES_OF_INTEREST.keys())
    
    client = NWCCClient()
    date_str = target_date.strftime("%Y-%m-%d")
    
    all_results = []
    
    for state in states:
        print(f"Fetching {state} SNOTEL stations...")
        stations = client.get_stations(state=state, network="SNTL")
        stations_dict = {s["stationTriplet"]: s for s in stations}
        triplets = list(stations_dict.keys())
        
        print(f"  Found {len(triplets)} stations, fetching data...")
        data = client.get_data(
            station_triplets=triplets,
            elements=["WTEQ", "SNWD"],
            begin_date=date_str,
            end_date=date_str,
            include_median=True,
            batch_size=15,
        )
        
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
            
            basin_index = None
            if wteq is not None and wteq_med is not None and wteq_med > 0:
                basin_index = (wteq / wteq_med) * 100
            elif wteq == 0:
                basin_index = 0
            
            all_results.append({
                "triplet": triplet,
                "name": station_info.get("name", triplet),
                "state_code": state,
                "latitude": lat,
                "longitude": lon,
                "elevation": station_info.get("elevation", 0),
                "huc": station_info.get("huc", ""),
                "wteq_value": wteq,
                "wteq_median": wteq_med,
                "basin_index": basin_index,
                "snow_depth": snwd,
            })
    
    if not all_results:
        return gpd.GeoDataFrame()
    
    df = pd.DataFrame(all_results)
    geometry = [Point(row.longitude, row.latitude) for row in df.itertuples()]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    print(f"Retrieved {len(gdf)} stations total")
    return gdf


def load_huc12_watersheds(
    states: list[str] = None,
    bbox: tuple = None,
    max_watersheds: int = 2000,
) -> gpd.GeoDataFrame:
    """Load HUC-12 watersheds, optionally filtered."""
    wbd_path = RAW_DATA_DIR / "watershed/national/WBD_National_GDB/WBD_National_GDB.gdb"
    
    if not wbd_path.exists():
        raise FileNotFoundError(f"WBD geodatabase not found at {wbd_path}")
    
    print("Loading HUC-12 watersheds...")
    
    if bbox:
        watersheds = gpd.read_file(wbd_path, layer="WBDHU12", bbox=bbox)
        print(f"  Loaded {len(watersheds)} watersheds in bounding box")
    else:
        watersheds = gpd.read_file(wbd_path, layer="WBDHU12")
    
    if states:
        state_huc_prefixes = {
            "CA": ["18"],
            "ID": ["17"],
            "NV": ["16"],
            "OR": ["17"],
            "WA": ["17"],
        }
        
        prefixes = []
        for state in states:
            prefixes.extend(state_huc_prefixes.get(state.upper(), []))
        
        if prefixes:
            prefixes = list(set(prefixes))
            mask = watersheds["huc12"].str[:2].isin(prefixes)
            watersheds = watersheds[mask]
            print(f"  Filtered to {len(watersheds)} watersheds")
    
    if len(watersheds) > max_watersheds:
        print(f"  Limiting to {max_watersheds} watersheds for performance")
        watersheds = watersheds.head(max_watersheds)
    
    return watersheds


def load_state_boundaries(states: list[str]) -> gpd.GeoDataFrame:
    """Load state boundaries."""
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
    target_date: date = None,
    output_path: str = None,
) -> folium.Map:
    """
    Create an interactive map showing snowpack conditions.
    """
    if snotel_gdf.empty:
        print("No data to map!")
        return None
    
    # Calculate map center
    center_lat = snotel_gdf.geometry.y.mean()
    center_lon = snotel_gdf.geometry.x.mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
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
    if watersheds_gdf is not None and not watersheds_gdf.empty:
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
    
    # Create color map (red to green, 0-100 scale)
    colormap = LinearColormap(
        colors=["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"],
        vmin=0,
        vmax=100,
        caption="Basin Index (% of Median)"
    )
    colormap.add_to(m)
    
    # Add SNOTEL stations as colored markers
    for idx, row in snotel_gdf.iterrows():
        basin_index = row.get("basin_index")
        
        if basin_index is None or pd.isna(basin_index):
            color = "#808080"
            basin_str = "N/A"
        else:
            clamped = max(0, min(150, basin_index))
            color = colormap(min(clamped, 100))
            basin_str = f"{basin_index:.0f}%"
        
        # Format values for popup
        wteq_str = f"{row.get('wteq_value', 0):.1f}\"" if row.get('wteq_value') is not None else "N/A"
        median_str = f"{row.get('wteq_median', 0):.1f}\"" if row.get('wteq_median') is not None else "N/A"
        snwd_str = f"{row.get('snow_depth', 0):.0f}\"" if row.get('snow_depth') is not None else "N/A"
        elev = row.get('elevation', 0) or 0
        
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 200px;">
            <h4 style="margin: 0 0 8px 0; color: #333;">{row.get('name', 'Unknown')}</h4>
            <table style="width: 100%; font-size: 12px;">
                <tr><td><b>Basin Index:</b></td><td style="color: {color}; font-weight: bold;">{basin_str}</td></tr>
                <tr><td><b>WTEQ:</b></td><td>{wteq_str}</td></tr>
                <tr><td><b>Median:</b></td><td>{median_str}</td></tr>
                <tr><td><b>Snow Depth:</b></td><td>{snwd_str}</td></tr>
                <tr><td><b>Elevation:</b></td><td>{elev:.0f} ft</td></tr>
                <tr><td><b>HUC:</b></td><td>{row.get('huc', 'N/A')}</td></tr>
            </table>
        </div>
        """
        
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=10,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row.get('name', 'Unknown')}: {basin_str}",
            color="#333333",
            weight=1,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
        ).add_to(m)
    
    # Add title
    date_str = target_date.strftime('%Y-%m-%d') if target_date else "Latest"
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px 20px; border-radius: 5px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial, sans-serif;">
        <h3 style="margin: 0; color: #333;">{title}</h3>
        <p style="margin: 5px 0 0 0; font-size: 12px; color: #666;">
            Date: {date_str} | Stations: {len(snotel_gdf)}
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    if output_path:
        m.save(output_path)
        print(f"Map saved to: {output_path}")
    
    return m


def main():
    parser = argparse.ArgumentParser(
        description="Create snowpack visualization map"
    )
    parser.add_argument(
        "--state",
        type=str,
        help="State code (e.g., OR, CA, WA)"
    )
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Include all states of interest"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Date to visualize (YYYY-MM-DD, default: latest in DB)"
    )
    parser.add_argument(
        "--no-watersheds",
        action="store_true",
        help="Skip loading HUC-12 watershed boundaries"
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Fetch data from API instead of database"
    )
    parser.add_argument(
        "--list-dates",
        action="store_true",
        help="List available dates in the database"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output HTML file path"
    )
    
    args = parser.parse_args()
    
    # List dates mode
    if args.list_dates:
        dates = get_available_dates()
        print(f"Available dates in database:")
        print(f"  Earliest: {dates['earliest']}")
        print(f"  Latest:   {dates['latest']}")
        print(f"  Total:    {dates['total_dates']} dates")
        return 0
    
    # Determine states
    if args.all_states:
        states = list(STATES_OF_INTEREST.keys())
    elif args.state:
        states = [args.state.upper()]
    else:
        print("Error: Specify --state or --all-states")
        parser.print_help()
        return 1
    
    # Determine date
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        dates = get_available_dates()
        target_date = dates['latest']
        print(f"Using latest date in database: {target_date}")
    
    # Fetch data
    if args.use_api:
        snotel_gdf = get_snotel_data_from_api(target_date, states)
    else:
        snotel_gdf = get_snotel_data_from_db(target_date, states)
    
    if snotel_gdf.empty:
        print("No data found!")
        return 1
    
    # Load watersheds
    watersheds_gdf = None
    if not args.no_watersheds:
        try:
            bounds = snotel_gdf.total_bounds
            buffer = 0.5
            bbox = (bounds[0] - buffer, bounds[1] - buffer, bounds[2] + buffer, bounds[3] + buffer)
            watersheds_gdf = load_huc12_watersheds(states=states, bbox=bbox)
        except FileNotFoundError as e:
            print(f"Warning: Could not load watersheds: {e}")
    
    # Load state boundaries
    try:
        state_boundaries = load_state_boundaries(states)
    except Exception as e:
        print(f"Warning: Could not load state boundaries: {e}")
        state_boundaries = None
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        state_str = "_".join(s.lower() for s in sorted(states))
        date_str = target_date.strftime("%Y%m%d")
        output_path = str(PROCESSED_DATA_DIR / f"snowpack_{state_str}_{date_str}.html")
    
    # Create map
    title = f"SNOTEL Basin Index - {', '.join(states)}"
    create_snowpack_map(
        snotel_gdf=snotel_gdf,
        watersheds_gdf=watersheds_gdf,
        state_boundaries=state_boundaries,
        title=title,
        target_date=target_date,
        output_path=output_path,
    )
    
    # Print summary
    valid_indices = snotel_gdf["basin_index"].dropna()
    if len(valid_indices) > 0:
        print(f"\nBasin Index Summary ({target_date}):")
        print(f"  Stations with data: {len(valid_indices)}")
        print(f"  Average: {valid_indices.mean():.0f}%")
        print(f"  Min: {valid_indices.min():.0f}%")
        print(f"  Max: {valid_indices.max():.0f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
