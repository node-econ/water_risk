#!/usr/bin/env python3
"""
SNOTEL Data Collection Script (Optimized)

Fetches SNOTEL data from the NWCC API and stores it in PostgreSQL.
Uses date range queries and concurrent requests for faster backfills.

Usage:
    # Sync stations (run first or periodically)
    python collect_snotel_data.py --sync-stations
    
    # Collect yesterday's data (daily cron job)
    python collect_snotel_data.py --collect
    
    # Backfill historical data (optimized)
    python collect_snotel_data.py --backfill --start 2024-10-01 --end 2025-09-30
    
    # Collect for specific states
    python collect_snotel_data.py --collect --states OR,WA
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime, timedelta, date, timezone
from typing import Optional

import aiohttp
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

from config import STATES_OF_INTEREST
from models import (
    Station, DailyObservation, DataFetchLog,
    get_session, get_engine
)


# API Configuration
API_BASE_URL = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
ELEMENTS = ["WTEQ", "SNWD", "PREC", "TMAX", "TMIN", "TAVG"]

# Tuning parameters - adjust these to balance speed vs server load
CONCURRENT_REQUESTS = 3      # Max concurrent API requests
STATIONS_PER_BATCH = 20      # Stations per API call
DAYS_PER_CHUNK = 30          # Days to fetch per API call (max ~90 recommended)
REQUEST_DELAY = 0.5          # Seconds between request batches
MAX_RETRIES = 3              # Retries for failed requests


def now_utc():
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


async def fetch_data_async(
    session: aiohttp.ClientSession,
    triplets: list[str],
    begin_date: str,
    end_date: str,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """
    Fetch data for a batch of stations over a date range.
    
    Args:
        session: aiohttp session
        triplets: List of station triplets
        begin_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        semaphore: Concurrency limiter
        
    Returns:
        List of station data dictionaries
    """
    params = {
        "stationTriplets": ",".join(triplets),
        "elements": ",".join(ELEMENTS),
        "duration": "DAILY",
        "beginDate": begin_date,
        "endDate": end_date,
        "centralTendencyType": "MEDIAN",
    }
    
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(
                    f"{API_BASE_URL}/data",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 500:
                        # Server error - wait and retry
                        await asyncio.sleep(2 ** attempt)
                    else:
                        print(f"    Warning: HTTP {response.status} for batch")
                        return []
            except asyncio.TimeoutError:
                print(f"    Timeout on attempt {attempt + 1}")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                print(f"    Error on attempt {attempt + 1}: {e}")
                await asyncio.sleep(1)
        
        return []


async def fetch_date_range_async(
    triplets: list[str],
    begin_date: date,
    end_date: date,
) -> list[dict]:
    """
    Fetch data for all stations over a date range using concurrent requests.
    
    Args:
        triplets: All station triplets to fetch
        begin_date: Start date
        end_date: End date
        
    Returns:
        Combined list of all station data
    """
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    # Create batches of stations
    station_batches = [
        triplets[i:i + STATIONS_PER_BATCH]
        for i in range(0, len(triplets), STATIONS_PER_BATCH)
    ]
    
    begin_str = begin_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    all_data = []
    
    async with aiohttp.ClientSession(
        headers={"Accept": "application/json", "User-Agent": "WaterRiskApp/1.0"}
    ) as session:
        # Process station batches with concurrency
        tasks = []
        for batch in station_batches:
            task = fetch_data_async(session, batch, begin_str, end_str, semaphore)
            tasks.append(task)
        
        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)
            elif isinstance(result, Exception):
                print(f"    Batch error: {result}")
        
        # Small delay between batches to be nice to the server
        await asyncio.sleep(REQUEST_DELAY)
    
    return all_data


def process_api_data(
    api_data: list[dict],
    stations_dict: dict,
    db_session,
) -> dict:
    """
    Process API response data and insert into database.
    
    Args:
        api_data: List of station data from API
        stations_dict: Dict mapping triplet -> Station object
        db_session: Database session
        
    Returns:
        Stats dictionary
    """
    stats = {"observations": 0, "stations": set()}
    
    observations_to_insert = []
    
    for station_data in api_data:
        triplet = station_data.get("stationTriplet", "")
        db_station = stations_dict.get(triplet)
        
        if not db_station:
            continue
        
        # Process each element's data
        element_data = {}
        for elem_data in station_data.get("data", []):
            elem_code = elem_data.get("stationElement", {}).get("elementCode", "")
            values = elem_data.get("values", [])
            
            for val_entry in values:
                date_str = val_entry.get("date", "")[:10]
                if not date_str:
                    continue
                
                try:
                    obs_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                except ValueError:
                    continue
                
                val = val_entry.get("value")
                median = val_entry.get("median")
                
                if val is None or val == -9999.9:
                    continue
                
                # Group by date
                if obs_date not in element_data:
                    element_data[obs_date] = {}
                
                element_data[obs_date][elem_code] = {"value": val, "median": median}
        
        # Create observation records
        for obs_date, elements in element_data.items():
            obs = {
                "station_id": db_station.id,
                "observation_date": obs_date,
                "fetched_at": now_utc(),
            }
            
            # Extract element values
            if "WTEQ" in elements:
                obs["wteq_value"] = elements["WTEQ"]["value"]
                obs["wteq_median"] = elements["WTEQ"]["median"]
                if obs["wteq_median"] and obs["wteq_median"] > 0:
                    obs["wteq_pct_median"] = (obs["wteq_value"] / obs["wteq_median"]) * 100
            
            if "SNWD" in elements:
                obs["snwd_value"] = elements["SNWD"]["value"]
                obs["snwd_median"] = elements["SNWD"]["median"]
                if obs["snwd_median"] and obs["snwd_median"] > 0:
                    obs["snwd_pct_median"] = (obs["snwd_value"] / obs["snwd_median"]) * 100
            
            if "PREC" in elements:
                obs["prec_value"] = elements["PREC"]["value"]
                obs["prec_median"] = elements["PREC"]["median"]
                if obs["prec_median"] and obs["prec_median"] > 0:
                    obs["prec_pct_median"] = (obs["prec_value"] / obs["prec_median"]) * 100
            
            if "TMAX" in elements:
                obs["tmax_value"] = elements["TMAX"]["value"]
            if "TMIN" in elements:
                obs["tmin_value"] = elements["TMIN"]["value"]
            if "TAVG" in elements:
                obs["tavg_value"] = elements["TAVG"]["value"]
            
            observations_to_insert.append(obs)
            stats["stations"].add(triplet)
    
    # Bulk upsert
    if observations_to_insert:
        for obs in observations_to_insert:
            stmt = insert(DailyObservation).values(**obs)
            stmt = stmt.on_conflict_do_update(
                constraint="uix_station_date",
                set_={k: v for k, v in obs.items() 
                      if k not in ("station_id", "observation_date")}
            )
            db_session.execute(stmt)
        
        db_session.commit()
        stats["observations"] = len(observations_to_insert)
    
    return stats


def sync_stations(states: list[str] = None, network: str = "SNTL") -> int:
    """Sync station metadata from NWCC API to database."""
    import requests
    
    if states is None:
        states = list(STATES_OF_INTEREST.keys())
    
    session = get_session()
    total_synced = 0
    
    for state in states:
        print(f"Syncing {state} {network} stations...")
        
        try:
            response = requests.get(
                f"{API_BASE_URL}/stations",
                params={"stationTriplets": f"*:{state}:{network}", "activeOnly": "false"},
                timeout=60
            )
            response.raise_for_status()
            api_stations = response.json()
            
            for api_station in api_stations:
                triplet = api_station.get("stationTriplet")
                if not triplet:
                    continue
                
                # Parse dates
                begin_date = end_date = None
                if api_station.get("beginDate"):
                    try:
                        begin_date = datetime.strptime(
                            api_station["beginDate"].split()[0], "%Y-%m-%d"
                        ).date()
                    except (ValueError, IndexError):
                        pass
                if api_station.get("endDate"):
                    try:
                        end_date = datetime.strptime(
                            api_station["endDate"].split()[0], "%Y-%m-%d"
                        ).date()
                    except (ValueError, IndexError):
                        pass
                
                is_active = end_date is None or end_date > date.today()
                
                stmt = insert(Station).values(
                    triplet=triplet,
                    station_id=api_station.get("stationId", ""),
                    state_code=api_station.get("stateCode", state),
                    network_code=api_station.get("networkCode", network),
                    name=api_station.get("name", triplet),
                    county_name=api_station.get("countyName"),
                    huc=api_station.get("huc"),
                    latitude=api_station.get("latitude", 0),
                    longitude=api_station.get("longitude", 0),
                    elevation=api_station.get("elevation"),
                    shef_id=api_station.get("shefId"),
                    dco_code=api_station.get("dcoCode"),
                    operator=api_station.get("operator"),
                    data_timezone=api_station.get("dataTimeZone"),
                    begin_date=begin_date,
                    end_date=end_date,
                    is_active=is_active,
                    updated_at=now_utc(),
                )
                
                stmt = stmt.on_conflict_do_update(
                    index_elements=["triplet"],
                    set_={
                        "name": stmt.excluded.name,
                        "county_name": stmt.excluded.county_name,
                        "huc": stmt.excluded.huc,
                        "latitude": stmt.excluded.latitude,
                        "longitude": stmt.excluded.longitude,
                        "elevation": stmt.excluded.elevation,
                        "begin_date": stmt.excluded.begin_date,
                        "end_date": stmt.excluded.end_date,
                        "is_active": stmt.excluded.is_active,
                        "updated_at": stmt.excluded.updated_at,
                    }
                )
                
                session.execute(stmt)
                total_synced += 1
            
            session.commit()
            print(f"  Synced {len(api_stations)} stations for {state}")
            
        except Exception as e:
            print(f"  Error syncing {state}: {e}")
            session.rollback()
    
    session.close()
    print(f"\nTotal stations synced: {total_synced}")
    return total_synced


def backfill_data_optimized(
    start_date: date,
    end_date: date,
    states: list[str] = None,
    network: str = "SNTL",
    chunk_days: int = DAYS_PER_CHUNK,
    concurrent: int = CONCURRENT_REQUESTS,
) -> None:
    """
    Backfill historical data using optimized chunked requests.
    
    Fetches data in chunk_days day chunks instead of day-by-day.
    """
    if states is None:
        states = list(STATES_OF_INTEREST.keys())
    
    db_session = get_session()
    
    # Get all stations
    db_stations = db_session.query(Station).filter(
        Station.state_code.in_(states),
        Station.network_code == network,
    ).all()
    
    stations_dict = {s.triplet: s for s in db_stations}
    triplets = list(stations_dict.keys())
    
    total_days = (end_date - start_date).days + 1
    
    print(f"=" * 60)
    print(f"SNOTEL Backfill - Optimized")
    print(f"=" * 60)
    print(f"States: {', '.join(states)}")
    print(f"Stations: {len(triplets)}")
    print(f"Date range: {start_date} to {end_date} ({total_days} days)")
    print(f"Chunk size: {chunk_days} days")
    print(f"Concurrent requests: {concurrent}")
    print(f"=" * 60)
    print()
    
    # Process in chunks
    chunk_start = start_date
    chunk_num = 0
    total_observations = 0
    start_time = time.time()
    
    while chunk_start <= end_date:
        chunk_num += 1
        chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), end_date)
        chunk_days = (chunk_end - chunk_start).days + 1
        
        print(f"Chunk {chunk_num}: {chunk_start} to {chunk_end} ({chunk_days} days)")
        
        # Fetch data asynchronously
        chunk_start_time = time.time()
        api_data = asyncio.run(fetch_date_range_async(triplets, chunk_start, chunk_end))
        fetch_time = time.time() - chunk_start_time
        
        # Process and insert
        insert_start_time = time.time()
        stats = process_api_data(api_data, stations_dict, db_session)
        insert_time = time.time() - insert_start_time
        
        total_observations += stats["observations"]
        
        print(f"  Fetched: {len(api_data)} station records in {fetch_time:.1f}s")
        print(f"  Inserted: {stats['observations']} observations in {insert_time:.1f}s")
        print(f"  Stations with data: {len(stats['stations'])}")
        
        # Progress estimate
        elapsed = time.time() - start_time
        days_processed = (chunk_end - start_date).days + 1
        if days_processed > 0:
            rate = elapsed / days_processed
            remaining_days = (end_date - chunk_end).days
            eta_seconds = rate * remaining_days
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            print(f"  Progress: {days_processed}/{total_days} days | ETA: {eta_str}")
        
        print()
        
        chunk_start = chunk_end + timedelta(days=1)
        
        # Small delay between chunks
        time.sleep(REQUEST_DELAY)
    
    elapsed = time.time() - start_time
    
    print(f"=" * 60)
    print(f"Backfill Complete!")
    print(f"=" * 60)
    print(f"Total observations: {total_observations:,}")
    print(f"Total time: {timedelta(seconds=int(elapsed))}")
    print(f"Rate: {total_observations / elapsed:.1f} obs/sec")
    
    db_session.close()


def collect_daily_data(
    target_date: date,
    states: list[str] = None,
    network: str = "SNTL",
) -> dict:
    """Collect data for a single day (for daily cron jobs)."""
    if states is None:
        states = list(STATES_OF_INTEREST.keys())
    
    db_session = get_session()
    
    # Get stations
    db_stations = db_session.query(Station).filter(
        Station.state_code.in_(states),
        Station.network_code == network,
        Station.is_active == True,
    ).all()
    
    stations_dict = {s.triplet: s for s in db_stations}
    triplets = list(stations_dict.keys())
    
    print(f"Collecting data for {target_date} ({len(triplets)} stations)...")
    
    # Fetch data
    api_data = asyncio.run(fetch_date_range_async(triplets, target_date, target_date))
    
    # Process
    stats = process_api_data(api_data, stations_dict, db_session)
    
    db_session.close()
    
    return {
        "stations_queried": len(triplets),
        "stations_with_data": len(stats["stations"]),
        "records_inserted": stats["observations"],
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SNOTEL Data Collection")
    
    parser.add_argument("--sync-stations", action="store_true", help="Sync station metadata")
    parser.add_argument("--collect", action="store_true", help="Collect daily data")
    parser.add_argument("--backfill", action="store_true", help="Backfill historical data")
    
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--start", type=str, help="Start date for backfill")
    parser.add_argument("--end", type=str, help="End date for backfill")
    parser.add_argument("--states", type=str, help="Comma-separated state codes")
    parser.add_argument("--network", type=str, default="SNTL", help="Network code")
    
    # Tuning options
    parser.add_argument("--chunk-days", type=int, default=DAYS_PER_CHUNK,
                        help=f"Days per API chunk (default: {DAYS_PER_CHUNK})")
    parser.add_argument("--concurrent", type=int, default=CONCURRENT_REQUESTS,
                        help=f"Concurrent requests (default: {CONCURRENT_REQUESTS})")
    
    args = parser.parse_args()
    
    states = None
    if args.states:
        states = [s.strip().upper() for s in args.states.split(",")]
    
    if args.sync_stations:
        sync_stations(states, args.network)
    
    elif args.collect:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today() - timedelta(days=1)
        stats = collect_daily_data(target_date, states, args.network)
        print(f"\nSummary: {stats['records_inserted']} records from {stats['stations_with_data']}/{stats['stations_queried']} stations")
    
    elif args.backfill:
        if not args.start or not args.end:
            print("Error: --backfill requires --start and --end dates")
            return 1
        
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
        backfill_data_optimized(
            start_date, end_date, states, args.network,
            chunk_days=args.chunk_days,
            concurrent=args.concurrent,
        )
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
