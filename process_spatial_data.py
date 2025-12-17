#!/usr/bin/env python3
"""
Spatial Data Processing for Water Risk Application

This module provides utilities to process and combine the downloaded spatial datasets.
It demonstrates how to:
- Load watershed boundaries (HUC-12) from geodatabase
- Filter to states of interest (CA, NV, OR, WA)
- Overlay political boundaries
- Export processed data to analysis-ready formats

Usage:
    python process_spatial_data.py --list-layers        # List available layers
    python process_spatial_data.py --extract-region     # Extract regional data
    python process_spatial_data.py --create-geopackage  # Create unified GeoPackage
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, STATES_OF_INTEREST


# Standard CRS for analysis
# Using Albers Equal Area for accurate area calculations in the western US
ANALYSIS_CRS = CRS.from_epsg(5070)  # NAD83 / Conus Albers
WEB_CRS = CRS.from_epsg(3857)       # Web Mercator for visualization
WGS84 = CRS.from_epsg(4326)         # WGS84 for interchange


class SpatialDataProcessor:
    """Process and combine spatial datasets for water risk analysis."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize processor with data directory paths."""
        self.raw_dir = data_dir or RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # State FIPS codes for filtering
        self.state_fips = [info["fips"] for info in STATES_OF_INTEREST.values()]
        self.state_abbrevs = list(STATES_OF_INTEREST.keys())

    def list_geodatabase_layers(self, gdb_path: Path) -> list[str]:
        """List all layers in a geodatabase."""
        import fiona
        return fiona.listlayers(str(gdb_path))

    def load_states(self) -> gpd.GeoDataFrame:
        """Load state boundaries and filter to states of interest."""
        states_path = self.raw_dir / "political/national/tl_2023_us_state"
        shp_file = list(states_path.glob("*.shp"))[0]
        
        states = gpd.read_file(shp_file)
        # Filter to states of interest using FIPS codes
        states_filtered = states[states["STATEFP"].isin(self.state_fips)]
        
        print(f"Loaded {len(states_filtered)} states: {', '.join(states_filtered['STUSPS'].tolist())}")
        return states_filtered

    def load_counties(self, states_gdf: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
        """Load county boundaries, optionally filtered to states of interest."""
        counties_path = self.raw_dir / "political/national/tl_2023_us_county"
        shp_file = list(counties_path.glob("*.shp"))[0]
        
        counties = gpd.read_file(shp_file)
        
        if states_gdf is not None:
            # Filter by state FIPS
            counties = counties[counties["STATEFP"].isin(self.state_fips)]
        
        print(f"Loaded {len(counties)} counties")
        return counties

    def load_watersheds_huc12(
        self, 
        region_boundary: Optional[gpd.GeoDataFrame] = None,
        layer: str = "WBDHU12"
    ) -> gpd.GeoDataFrame:
        """
        Load HUC-12 watershed boundaries from WBD geodatabase.
        
        Args:
            region_boundary: Optional GeoDataFrame to clip watersheds to
            layer: Layer name (default WBDHU12 for HUC-12)
            
        Returns:
            GeoDataFrame of watershed boundaries
        """
        wbd_path = self.raw_dir / "watershed/national/WBD_National_GDB/WBD_National_GDB.gdb"
        
        if not wbd_path.exists():
            raise FileNotFoundError(f"WBD geodatabase not found at {wbd_path}")
        
        print(f"Loading {layer} from WBD geodatabase...")
        print("  (This may take a few minutes for the full national dataset)")
        
        # Load the layer
        watersheds = gpd.read_file(wbd_path, layer=layer)
        print(f"  Loaded {len(watersheds)} total watersheds")
        
        if region_boundary is not None:
            # Ensure same CRS
            if watersheds.crs != region_boundary.crs:
                region_boundary = region_boundary.to_crs(watersheds.crs)
            
            # Create unified region boundary
            region_union = region_boundary.union_all()
            
            # Clip watersheds to region
            print("  Clipping to region of interest...")
            watersheds = gpd.clip(watersheds, region_union)
            print(f"  {len(watersheds)} watersheds after clipping")
        
        return watersheds

    def load_congressional_districts(self) -> gpd.GeoDataFrame:
        """Load congressional district boundaries."""
        cd_path = self.raw_dir / "political/national/cb_2023_us_cd118_500k"
        shp_file = list(cd_path.glob("*.shp"))[0]
        
        districts = gpd.read_file(shp_file)
        # Filter to states of interest
        districts = districts[districts["STATEFP"].isin(self.state_fips)]
        
        print(f"Loaded {len(districts)} congressional districts")
        return districts

    def load_places(self, state: str) -> gpd.GeoDataFrame:
        """Load places (cities/towns) for a specific state."""
        state_lower = state.lower()
        places_path = self.raw_dir / f"political/{state_lower}"
        
        # Find place shapefile
        place_files = list(places_path.glob("*_place/*.shp"))
        if not place_files:
            raise FileNotFoundError(f"No place shapefile found for {state}")
        
        places = gpd.read_file(place_files[0])
        print(f"Loaded {len(places)} places for {state}")
        return places

    def load_tribal_areas(self) -> gpd.GeoDataFrame:
        """Load tribal area boundaries."""
        tribal_path = self.raw_dir / "political/national/tl_2023_us_aiannh"
        shp_file = list(tribal_path.glob("*.shp"))[0]
        
        tribal = gpd.read_file(shp_file)
        print(f"Loaded {len(tribal)} tribal areas")
        return tribal

    def load_urban_areas(self) -> gpd.GeoDataFrame:
        """Load urban area boundaries."""
        urban_path = self.raw_dir / "political/national/cb_2020_us_ua20_500k"
        shp_file = list(urban_path.glob("*.shp"))[0]
        
        urban = gpd.read_file(shp_file)
        print(f"Loaded {len(urban)} urban areas")
        return urban

    def load_state_legislative_districts(
        self, 
        state: str, 
        chamber: str = "both"
    ) -> gpd.GeoDataFrame:
        """
        Load state legislative district boundaries.
        
        Args:
            state: State abbreviation (CA, NV, OR, WA)
            chamber: 'upper' (Senate), 'lower' (Assembly/House), or 'both'
            
        Returns:
            GeoDataFrame of legislative districts
        """
        state_upper = state.upper()
        state_lower = state.lower()
        
        if state_upper not in STATES_OF_INTEREST:
            raise ValueError(f"State {state} not in states of interest")
        
        fips = STATES_OF_INTEREST[state_upper]["fips"]
        state_path = self.raw_dir / f"political/{state_lower}"
        
        results = []
        
        if chamber in ("upper", "both"):
            # Upper chamber (Senate)
            upper_path = state_path / f"tl_2023_{fips}_sldu"
            if upper_path.exists():
                shp_files = list(upper_path.glob("*.shp"))
                if shp_files:
                    upper = gpd.read_file(shp_files[0])
                    upper["chamber"] = "upper"
                    upper["chamber_name"] = "Senate"
                    results.append(upper)
                    print(f"Loaded {len(upper)} {state} Senate districts")
        
        if chamber in ("lower", "both"):
            # Lower chamber (Assembly/House)
            lower_path = state_path / f"tl_2023_{fips}_sldl"
            if lower_path.exists():
                shp_files = list(lower_path.glob("*.shp"))
                if shp_files:
                    lower = gpd.read_file(shp_files[0])
                    lower["chamber"] = "lower"
                    # Different names by state
                    chamber_names = {
                        "CA": "Assembly", "NV": "Assembly",
                        "OR": "House", "WA": "House"
                    }
                    lower["chamber_name"] = chamber_names.get(state_upper, "House")
                    results.append(lower)
                    print(f"Loaded {len(lower)} {state} {lower['chamber_name'].iloc[0]} districts")
        
        if not results:
            raise FileNotFoundError(f"No legislative districts found for {state}")
        
        # Combine if both chambers requested
        if len(results) == 2:
            combined = pd.concat(results, ignore_index=True)
            return gpd.GeoDataFrame(combined, crs=results[0].crs)
        
        return results[0]

    def load_all_state_legislative_districts(self) -> gpd.GeoDataFrame:
        """Load all state legislative districts for all states of interest."""
        all_districts = []
        
        for state in self.state_abbrevs:
            try:
                districts = self.load_state_legislative_districts(state, chamber="both")
                districts["state"] = state
                all_districts.append(districts)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
        
        if not all_districts:
            raise FileNotFoundError("No state legislative districts found")
        
        combined = pd.concat(all_districts, ignore_index=True)
        gdf = gpd.GeoDataFrame(combined, crs=all_districts[0].crs)
        print(f"Total: {len(gdf)} state legislative districts across {len(all_districts)} states")
        return gdf

    def load_padus(
        self, 
        state: str, 
        layer_type: str = "fee"
    ) -> gpd.GeoDataFrame:
        """
        Load PAD-US 4.1 protected areas for a specific state.
        
        Args:
            state: State abbreviation (CA, NV, OR, WA)
            layer_type: Type of protected area layer:
                - 'fee': Fee simple ownership (BLM, NPS, USFS, state parks, etc.)
                - 'easement': Conservation easements
                - 'proclamation': Proclaimed boundaries (National Monuments, etc.)
                - 'designation': Designated areas (Wilderness, WSAs, etc.)
                - 'combined': Combined/dissolved layer for analysis
                - 'marine': Marine protected areas (coastal states only)
                
        Returns:
            GeoDataFrame of protected areas
        """
        state_upper = state.upper()
        state_lower = state.lower()
        
        if state_upper not in STATES_OF_INTEREST:
            raise ValueError(f"State {state} not in states of interest")
        
        # Find the geodatabase
        gdb_path = self.raw_dir / f"protected/{state_lower}/PADUS4_1_State{state_upper}.gdb"
        
        if not gdb_path.exists():
            raise FileNotFoundError(f"PAD-US geodatabase not found at {gdb_path}")
        
        # Map layer_type to actual layer name
        layer_map = {
            "fee": f"PADUS4_1Fee_State_{state_upper}",
            "easement": f"PADUS4_1Easement_State_{state_upper}",
            "proclamation": f"PADUS4_1Proclamation_State_{state_upper}",
            "designation": f"PADUS4_1Designation_State_{state_upper}",
            "combined": f"PADUS4_1Comb_DOD_Trib_NGP_Fee_Desig_Ease_State_{state_upper}",
            "marine": "PADUS4_1Marine",
        }
        
        if layer_type not in layer_map:
            raise ValueError(f"Invalid layer_type: {layer_type}. Choose from: {list(layer_map.keys())}")
        
        layer_name = layer_map[layer_type]
        
        # Check if layer exists
        import fiona
        available_layers = fiona.listlayers(str(gdb_path))
        
        if layer_name not in available_layers:
            raise ValueError(f"Layer '{layer_name}' not found in {gdb_path}. Available: {available_layers}")
        
        print(f"Loading PAD-US {layer_type} layer for {state_upper}...")
        gdf = gpd.read_file(gdb_path, layer=layer_name)
        print(f"  Loaded {len(gdf)} protected areas")
        
        return gdf

    def load_padus_all_states(
        self, 
        layer_type: str = "fee"
    ) -> gpd.GeoDataFrame:
        """
        Load PAD-US 4.1 protected areas for all states of interest.
        
        Args:
            layer_type: Type of protected area layer (fee, easement, proclamation, 
                       designation, combined, marine)
                       
        Returns:
            Combined GeoDataFrame of protected areas for all states
        """
        all_areas = []
        
        for state in self.state_abbrevs:
            try:
                gdf = self.load_padus(state, layer_type=layer_type)
                gdf["source_state"] = state
                all_areas.append(gdf)
            except (FileNotFoundError, ValueError) as e:
                print(f"  Warning: {e}")
        
        if not all_areas:
            raise FileNotFoundError(f"No PAD-US {layer_type} data found for any state")
        
        # Combine all states
        combined = pd.concat(all_areas, ignore_index=True)
        gdf = gpd.GeoDataFrame(combined, crs=all_areas[0].crs)
        print(f"Total: {len(gdf)} {layer_type} protected areas across {len(all_areas)} states")
        
        return gdf

    def load_padus_comprehensive(self) -> gpd.GeoDataFrame:
        """
        Load all PAD-US fee ownership areas for all states.
        
        This is the most commonly used layer, containing all public lands
        owned in fee simple (national parks, forests, BLM lands, state parks, etc.)
        
        Returns:
            Combined GeoDataFrame of all fee ownership protected areas
        """
        return self.load_padus_all_states(layer_type="fee")

    def list_padus_layers(self, state: str) -> list[str]:
        """List available layers in a state's PAD-US geodatabase."""
        state_upper = state.upper()
        state_lower = state.lower()
        
        gdb_path = self.raw_dir / f"protected/{state_lower}/PADUS4_1_State{state_upper}.gdb"
        
        if not gdb_path.exists():
            raise FileNotFoundError(f"PAD-US geodatabase not found at {gdb_path}")
        
        import fiona
        return fiona.listlayers(str(gdb_path))

    def create_regional_watershed_layer(self, output_format: str = "gpkg") -> Path:
        """
        Create a regional HUC-12 watershed layer clipped to states of interest.
        
        Args:
            output_format: Output format ('gpkg', 'shp', or 'geojson')
            
        Returns:
            Path to output file
        """
        # Load states and create region boundary
        states = self.load_states()
        
        # Load and clip watersheds
        watersheds = self.load_watersheds_huc12(region_boundary=states)
        
        # Reproject to analysis CRS
        print("Reprojecting to analysis CRS (NAD83 / Conus Albers)...")
        watersheds = watersheds.to_crs(ANALYSIS_CRS)
        
        # Calculate area in square kilometers
        watersheds["area_km2"] = watersheds.geometry.area / 1e6
        
        # Save output
        output_name = f"watersheds_huc12_west_coast.{output_format}"
        output_path = self.processed_dir / output_name
        
        print(f"Saving to {output_path}...")
        if output_format == "gpkg":
            watersheds.to_file(output_path, driver="GPKG", layer="huc12")
        elif output_format == "shp":
            watersheds.to_file(output_path)
        elif output_format == "geojson":
            watersheds.to_file(output_path, driver="GeoJSON")
        
        print(f"✓ Created regional watershed layer: {output_path}")
        print(f"  Total watersheds: {len(watersheds)}")
        print(f"  Total area: {watersheds['area_km2'].sum():,.0f} km²")
        
        return output_path

    def create_unified_geopackage(self) -> Path:
        """
        Create a unified GeoPackage with all regional layers.
        
        Includes:
        - HUC-12 watersheds
        - State boundaries
        - County boundaries
        - Congressional districts
        - Places (all states combined)
        - Tribal areas (clipped to region)
        - Urban areas (clipped to region)
        """
        output_path = self.processed_dir / "water_risk_boundaries.gpkg"
        
        # Load states first (used for clipping)
        print("\n=== Creating Unified GeoPackage ===\n")
        states = self.load_states()
        states_albers = states.to_crs(ANALYSIS_CRS)
        region_union = states_albers.union_all()
        
        # Save states
        print("Saving states layer...")
        states_albers.to_file(output_path, driver="GPKG", layer="states")
        
        # Counties
        print("Processing counties...")
        counties = self.load_counties(states)
        counties = counties.to_crs(ANALYSIS_CRS)
        counties.to_file(output_path, driver="GPKG", layer="counties")
        
        # Congressional districts
        print("Processing congressional districts...")
        districts = self.load_congressional_districts()
        districts = districts.to_crs(ANALYSIS_CRS)
        districts.to_file(output_path, driver="GPKG", layer="congressional_districts")
        
        # Places - combine all states
        print("Processing places...")
        all_places = []
        for state in self.state_abbrevs:
            try:
                places = self.load_places(state)
                places["STATE"] = state
                all_places.append(places)
            except FileNotFoundError:
                print(f"  Warning: No places found for {state}")
        
        if all_places:
            places_combined = pd.concat(all_places, ignore_index=True)
            places_gdf = gpd.GeoDataFrame(places_combined, crs=all_places[0].crs)
            places_gdf = places_gdf.to_crs(ANALYSIS_CRS)
            places_gdf.to_file(output_path, driver="GPKG", layer="places")
        
        # Tribal areas - clip to region
        print("Processing tribal areas...")
        tribal = self.load_tribal_areas()
        tribal = tribal.to_crs(ANALYSIS_CRS)
        tribal = gpd.clip(tribal, region_union)
        if len(tribal) > 0:
            tribal.to_file(output_path, driver="GPKG", layer="tribal_areas")
        
        # Urban areas - clip to region
        print("Processing urban areas...")
        urban = self.load_urban_areas()
        urban = urban.to_crs(ANALYSIS_CRS)
        urban = gpd.clip(urban, region_union)
        if len(urban) > 0:
            urban.to_file(output_path, driver="GPKG", layer="urban_areas")
        
        # State legislative districts
        print("Processing state legislative districts...")
        try:
            leg_districts = self.load_all_state_legislative_districts()
            leg_districts = leg_districts.to_crs(ANALYSIS_CRS)
            leg_districts.to_file(output_path, driver="GPKG", layer="state_legislative_districts")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
        
        # PAD-US Protected Areas
        print("Processing PAD-US protected areas...")
        try:
            # Fee ownership (most common layer)
            padus_fee = self.load_padus_all_states(layer_type="fee")
            padus_fee = padus_fee.to_crs(ANALYSIS_CRS)
            padus_fee["area_km2"] = padus_fee.geometry.area / 1e6
            padus_fee.to_file(output_path, driver="GPKG", layer="protected_areas_fee")
            
            # Easements
            padus_ease = self.load_padus_all_states(layer_type="easement")
            padus_ease = padus_ease.to_crs(ANALYSIS_CRS)
            padus_ease["area_km2"] = padus_ease.geometry.area / 1e6
            padus_ease.to_file(output_path, driver="GPKG", layer="protected_areas_easement")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
        
        # Watersheds (this is the big one)
        print("Processing HUC-12 watersheds (this may take several minutes)...")
        watersheds = self.load_watersheds_huc12(region_boundary=states)
        watersheds = watersheds.to_crs(ANALYSIS_CRS)
        watersheds["area_km2"] = watersheds.geometry.area / 1e6
        watersheds.to_file(output_path, driver="GPKG", layer="watersheds_huc12")
        
        print(f"\n✓ Created unified GeoPackage: {output_path}")
        
        # List layers
        import fiona
        layers = fiona.listlayers(str(output_path))
        print(f"  Layers: {', '.join(layers)}")
        
        return output_path

    def get_watershed_stats(self, watersheds: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate summary statistics for watersheds."""
        stats = {
            "total_count": len(watersheds),
            "total_area_km2": watersheds["area_km2"].sum() if "area_km2" in watersheds.columns else None,
            "mean_area_km2": watersheds["area_km2"].mean() if "area_km2" in watersheds.columns else None,
            "min_area_km2": watersheds["area_km2"].min() if "area_km2" in watersheds.columns else None,
            "max_area_km2": watersheds["area_km2"].max() if "area_km2" in watersheds.columns else None,
        }
        return pd.DataFrame([stats])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process spatial data for water risk analysis"
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="List available layers in WBD geodatabase"
    )
    parser.add_argument(
        "--extract-region",
        action="store_true",
        help="Extract HUC-12 watersheds for region of interest"
    )
    parser.add_argument(
        "--create-geopackage",
        action="store_true",
        help="Create unified GeoPackage with all layers"
    )
    parser.add_argument(
        "--output-format",
        choices=["gpkg", "shp", "geojson"],
        default="gpkg",
        help="Output format for extracted data (default: gpkg)"
    )
    
    args = parser.parse_args()
    processor = SpatialDataProcessor()
    
    if args.list_layers:
        wbd_path = RAW_DATA_DIR / "watershed/national/WBD_National_GDB/WBD_National_GDB.gdb"
        if wbd_path.exists():
            layers = processor.list_geodatabase_layers(wbd_path)
            print("Available layers in WBD geodatabase:")
            for layer in layers:
                print(f"  - {layer}")
        else:
            print(f"WBD geodatabase not found at {wbd_path}")
        return 0
    
    if args.extract_region:
        processor.create_regional_watershed_layer(output_format=args.output_format)
        return 0
    
    if args.create_geopackage:
        processor.create_unified_geopackage()
        return 0
    
    # Default: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

