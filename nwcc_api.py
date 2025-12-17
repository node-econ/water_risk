#!/usr/bin/env python3
"""
NWCC AWDB REST API Client

Client for the USDA National Water and Climate Center's 
Air-Water Database (AWDB) REST API.

API Documentation: https://wcc.sc.egov.usda.gov/awdbRestApi/swagger-ui/index.html

Available data includes:
- SNOTEL (Snow Telemetry) stations
- SCAN (Soil Climate Analysis Network) stations
- Snow courses
- Reservoir data
- Streamflow data

Example usage:
    python nwcc_api.py --state OR --report snotel
    python nwcc_api.py --state CA --network SNTL --elements WTEQ,SNWD
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from typing import Optional

import requests


class NWCCClient:
    """Client for the NWCC AWDB REST API."""
    
    BASE_URL = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
    
    # Common element codes
    ELEMENTS = {
        "WTEQ": "Snow Water Equivalent",
        "SNWD": "Snow Depth", 
        "PREC": "Precipitation Accumulation",
        "TAVG": "Air Temperature Average",
        "TMAX": "Air Temperature Maximum",
        "TMIN": "Air Temperature Minimum",
        "TOBS": "Air Temperature Observed",
        "SMS": "Soil Moisture",
        "STO": "Soil Temperature",
        "RESC": "Reservoir Storage",
        "SRDOX": "Streamflow Discharge",
    }
    
    # Network codes
    NETWORKS = {
        "SNTL": "SNOTEL",
        "SCAN": "SCAN",
        "SNOW": "Snow Course",
        "BOR": "Reservoir",
        "USGS": "Streamflow",
        "SNTLT": "SNOLITE",
    }
    
    def __init__(self, timeout: int = 60):
        """Initialize the client."""
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "WaterRiskApp/1.0"
        })
    
    def get_stations(
        self,
        state: Optional[str] = None,
        network: str = "SNTL",
        active_only: bool = True,
        hucs: Optional[str] = None,
    ) -> list[dict]:
        """
        Get station metadata.
        
        Args:
            state: Two-letter state code (e.g., 'OR', 'CA')
            network: Network code (SNTL, SCAN, SNOW, etc.)
            active_only: Only return active stations
            hucs: Filter by HUC codes (comma-separated, supports wildcards)
            
        Returns:
            List of station metadata dictionaries
        """
        params = {"activeOnly": str(active_only).lower()}
        
        if state:
            params["stationTriplets"] = f"*:{state}:{network}"
        
        if hucs:
            params["hucs"] = hucs
            
        response = self.session.get(
            f"{self.BASE_URL}/stations",
            params=params,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_data(
        self,
        station_triplets: list[str],
        elements: list[str] = ["WTEQ"],
        duration: str = "DAILY",
        begin_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_median: bool = True,
        batch_size: int = 10,
    ) -> list[dict]:
        """
        Get data for stations.
        
        Args:
            station_triplets: List of station triplets (e.g., ['302:OR:SNTL'])
            elements: List of element codes (e.g., ['WTEQ', 'SNWD'])
            duration: Data duration (DAILY, HOURLY, etc.)
            begin_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_median: Include median values for comparison
            batch_size: Number of stations per API request
            
        Returns:
            List of station data dictionaries
        """
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        if begin_date is None:
            begin_date = end_date
            
        all_data = []
        
        # Process in batches to avoid API limits
        for i in range(0, len(station_triplets), batch_size):
            batch = station_triplets[i:i + batch_size]
            
            params = {
                "stationTriplets": ",".join(batch),
                "elements": ",".join(elements),
                "duration": duration,
                "beginDate": begin_date,
                "endDate": end_date,
            }
            
            if include_median:
                params["centralTendencyType"] = "MEDIAN"
            
            try:
                response = self.session.get(
                    f"{self.BASE_URL}/data",
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                all_data.extend(response.json())
            except requests.exceptions.RequestException as e:
                print(f"Warning: Failed to fetch batch {i//batch_size + 1}: {e}")
                
        return all_data
    
    def get_forecasts(
        self,
        station_triplets: list[str],
        element_codes: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Get forecast data for stations.
        
        Args:
            station_triplets: List of station triplets
            element_codes: Element codes to retrieve forecasts for
            
        Returns:
            List of forecast data dictionaries
        """
        params = {"stationTriplets": ",".join(station_triplets)}
        
        if element_codes:
            params["elementCodes"] = ",".join(element_codes)
            
        response = self.session.get(
            f"{self.BASE_URL}/forecasts",
            params=params,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_reference_data(self, lists: list[str] = ["elements", "networks"]) -> dict:
        """Get reference data (elements, networks, etc.)."""
        params = {"referenceLists": ",".join(lists)}
        response = self.session.get(
            f"{self.BASE_URL}/reference-data",
            params=params,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()


def generate_snotel_report(
    client: NWCCClient,
    state: str,
    date: Optional[str] = None,
) -> None:
    """
    Generate a SNOTEL update report similar to the NWCC web report.
    
    Args:
        client: NWCCClient instance
        state: Two-letter state code
        date: Report date (default: yesterday)
    """
    if date is None:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
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
        batch_size=15,  # Smaller batches for reliability
    )
    
    # Process results
    results = []
    for station_data in data:
        triplet = station_data.get("stationTriplet", "")
        station_info = stations_dict.get(triplet, {})
        
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
        
        if wteq is not None:
            pct = (wteq / wteq_med * 100) if wteq_med and wteq_med > 0 else None
            results.append({
                "name": station_info.get("name", triplet)[:27],
                "elevation": station_info.get("elevation", 0),
                "latitude": station_info.get("latitude"),
                "longitude": station_info.get("longitude"),
                "huc": station_info.get("huc", ""),
                "wteq": wteq,
                "median": wteq_med,
                "pct_median": pct,
                "snow_depth": snwd,
            })
    
    # Sort and print report
    results.sort(key=lambda x: x["name"])
    
    print()
    print("=" * 95)
    print(f"{state} SNOTEL UPDATE REPORT")
    print(f"Report Date: {date}")
    print("=" * 95)
    print()
    print(f"{'Station Name':28} {'Elev':>6} {'WTEQ':>8} {'Median':>8} {'% Med':>7} {'Snow Depth':>10}")
    print(f"{'':28} {'(ft)':>6} {'(in)':>8} {'(in)':>8} {'':>7} {'(in)':>10}")
    print("-" * 95)
    
    for r in results:
        pct_str = f"{r['pct_median']:.0f}%" if r['pct_median'] else "N/A"
        med_str = f"{r['median']:.1f}" if r['median'] else "N/A"
        snwd_str = f"{r['snow_depth']:.0f}" if r['snow_depth'] is not None else "N/A"
        
        print(f"{r['name']:28} {r['elevation']:>6.0f} {r['wteq']:>8.1f} {med_str:>8} {pct_str:>7} {snwd_str:>10}")
    
    print("-" * 95)
    print(f"Stations with data: {len(results)} of {len(triplets)}")
    
    # Summary statistics
    valid_pcts = [r["pct_median"] for r in results if r["pct_median"]]
    if valid_pcts:
        avg_pct = sum(valid_pcts) / len(valid_pcts)
        min_pct = min(valid_pcts)
        max_pct = max(valid_pcts)
        print(f"Percent of Median - Avg: {avg_pct:.0f}%, Min: {min_pct:.0f}%, Max: {max_pct:.0f}%")
    
    total_wteq = sum(r["wteq"] for r in results if r["wteq"])
    total_med = sum(r["median"] for r in results if r["median"])
    if total_med > 0:
        basin_pct = total_wteq / total_med * 100
        print(f"Basin-wide % of Median: {basin_pct:.0f}%")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NWCC AWDB REST API Client"
    )
    parser.add_argument(
        "--state",
        type=str,
        default="OR",
        help="Two-letter state code (default: OR)"
    )
    parser.add_argument(
        "--network",
        type=str,
        default="SNTL",
        choices=["SNTL", "SCAN", "SNOW", "BOR", "USGS"],
        help="Station network (default: SNTL)"
    )
    parser.add_argument(
        "--report",
        type=str,
        choices=["snotel", "stations", "elements"],
        default="snotel",
        help="Report type to generate"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Report date (YYYY-MM-DD, default: yesterday)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    client = NWCCClient()
    
    if args.report == "elements":
        ref = client.get_reference_data(["elements", "networks", "durations"])
        if args.json:
            print(json.dumps(ref, indent=2))
        else:
            print("Available Elements:")
            for elem in ref.get("elements", []):
                print(f"  {elem['code']:12} {elem['name']}")
    
    elif args.report == "stations":
        stations = client.get_stations(state=args.state, network=args.network)
        if args.json:
            print(json.dumps(stations, indent=2))
        else:
            print(f"{args.state} {args.network} Stations ({len(stations)} total):")
            for s in stations:
                print(f"  {s['stationTriplet']:20} {s['name'][:30]:30} Elev: {s.get('elevation', 'N/A')}")
    
    elif args.report == "snotel":
        generate_snotel_report(client, args.state, args.date)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

