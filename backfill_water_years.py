#!/usr/bin/env python3
"""
Water Year Batch Backfill Script

Backfills SNOTEL data one water year at a time (Oct 1 - Sep 30).
Designed for long-running historical backfills with progress tracking.

Usage:
    # Backfill specific water years
    python backfill_water_years.py --start-wy 1996 --end-wy 1997
    
    # Backfill all historical data (1996-2024)
    python backfill_water_years.py --start-wy 1996 --end-wy 2024
    
    # Check progress
    python backfill_water_years.py --status

Water Year Convention:
    Water Year 1996 = October 1, 1995 through September 30, 1996
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, date
from pathlib import Path

# Import database connection for status checks
try:
    from sqlalchemy import text
    from models import get_session
    HAS_DB = True
except ImportError:
    HAS_DB = False


def water_year_dates(wy: int) -> tuple[date, date]:
    """
    Get start and end dates for a water year.
    
    Water Year N runs from October 1 of year N-1 to September 30 of year N.
    
    Args:
        wy: Water year (e.g., 1996)
        
    Returns:
        Tuple of (start_date, end_date)
    """
    start = date(wy - 1, 10, 1)
    end = date(wy, 9, 30)
    return start, end


def check_status():
    """Check the current status of data in the database."""
    if not HAS_DB:
        print("Database module not available. Run from project directory.")
        return
    
    session = get_session()
    
    # Overall stats
    result = session.execute(text("""
        SELECT 
            COUNT(*) as total_observations,
            MIN(observation_date) as earliest,
            MAX(observation_date) as latest,
            COUNT(DISTINCT station_id) as stations
        FROM daily_observations
    """)).fetchone()
    
    print("=" * 60)
    print("SNOTEL Data Status")
    print("=" * 60)
    print(f"Total Observations: {result[0]:,}")
    print(f"Date Range: {result[1]} to {result[2]}")
    print(f"Stations with Data: {result[3]}")
    print()
    
    # By water year
    result = session.execute(text("""
        SELECT 
            CASE 
                WHEN EXTRACT(MONTH FROM observation_date) >= 10 
                THEN EXTRACT(YEAR FROM observation_date) + 1
                ELSE EXTRACT(YEAR FROM observation_date)
            END as water_year,
            COUNT(*) as observations,
            COUNT(DISTINCT station_id) as stations
        FROM daily_observations
        GROUP BY water_year
        ORDER BY water_year
    """)).fetchall()
    
    print("By Water Year:")
    print(f"{'WY':>6} {'Observations':>15} {'Stations':>10}")
    print("-" * 35)
    for row in result:
        print(f"{int(row[0]):>6} {row[1]:>15,} {row[2]:>10}")
    
    # By state
    print()
    result = session.execute(text("""
        SELECT 
            s.state_code,
            COUNT(*) as observations,
            MIN(d.observation_date) as earliest,
            MAX(d.observation_date) as latest
        FROM daily_observations d
        JOIN stations s ON d.station_id = s.id
        GROUP BY s.state_code
        ORDER BY s.state_code
    """)).fetchall()
    
    print("By State:")
    print(f"{'State':>6} {'Observations':>15} {'Earliest':>12} {'Latest':>12}")
    print("-" * 50)
    for row in result:
        print(f"{row[0]:>6} {row[1]:>15,} {row[2]!s:>12} {row[3]!s:>12}")
    
    session.close()


def run_water_year_backfill(wy: int, verbose: bool = True) -> dict:
    """
    Run backfill for a single water year.
    
    Args:
        wy: Water year to backfill
        verbose: Print progress
        
    Returns:
        Dict with timing and status info
    """
    start_date, end_date = water_year_dates(wy)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Water Year {wy}")
        print(f"{'='*60}")
        print(f"Date range: {start_date} to {end_date}")
    
    start_time = time.time()
    
    # Run the collect script
    cmd = [
        "python", "collect_snotel_data.py",
        "--backfill",
        "--start", start_date.strftime("%Y-%m-%d"),
        "--end", end_date.strftime("%Y-%m-%d"),
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=not verbose,
        text=True,
    )
    
    elapsed = time.time() - start_time
    
    status = {
        "water_year": wy,
        "start_date": start_date,
        "end_date": end_date,
        "elapsed_seconds": elapsed,
        "success": result.returncode == 0,
    }
    
    if verbose:
        print(f"\nWater Year {wy} completed in {elapsed/60:.1f} minutes")
    
    return status


def run_batch_backfill(start_wy: int, end_wy: int, verbose: bool = True):
    """
    Run backfill for a range of water years.
    
    Args:
        start_wy: First water year
        end_wy: Last water year
        verbose: Print progress
    """
    total_years = end_wy - start_wy + 1
    
    print("=" * 60)
    print("SNOTEL Historical Backfill")
    print("=" * 60)
    print(f"Water Years: {start_wy} to {end_wy} ({total_years} years)")
    print(f"Date Range: {water_year_dates(start_wy)[0]} to {water_year_dates(end_wy)[1]}")
    print(f"Estimated time: {total_years * 7.5:.0f} minutes ({total_years * 7.5 / 60:.1f} hours)")
    print("=" * 60)
    
    results = []
    batch_start = time.time()
    
    for i, wy in enumerate(range(start_wy, end_wy + 1), 1):
        print(f"\n[{i}/{total_years}] Processing Water Year {wy}...")
        
        status = run_water_year_backfill(wy, verbose=verbose)
        results.append(status)
        
        # Progress estimate
        elapsed = time.time() - batch_start
        avg_per_year = elapsed / i
        remaining = (total_years - i) * avg_per_year
        
        print(f"Progress: {i}/{total_years} years complete")
        print(f"Elapsed: {elapsed/60:.1f} min | ETA: {remaining/60:.1f} min remaining")
    
    # Final summary
    total_elapsed = time.time() - batch_start
    successful = sum(1 for r in results if r["success"])
    
    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"Water Years: {start_wy} to {end_wy}")
    print(f"Successful: {successful}/{total_years}")
    print(f"Total Time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    print(f"Average: {total_elapsed/total_years/60:.1f} minutes per water year")
    
    # Check final status
    if HAS_DB:
        print()
        check_status()


def main():
    parser = argparse.ArgumentParser(
        description="Water Year Batch Backfill for SNOTEL Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test with two water years
    python backfill_water_years.py --start-wy 1996 --end-wy 1997
    
    # Full historical backfill
    python backfill_water_years.py --start-wy 1996 --end-wy 2024
    
    # Check current data status
    python backfill_water_years.py --status
        """
    )
    
    parser.add_argument(
        "--start-wy",
        type=int,
        help="Starting water year (e.g., 1996 = Oct 1995 - Sep 1996)"
    )
    parser.add_argument(
        "--end-wy", 
        type=int,
        help="Ending water year"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current data status"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    if args.status:
        check_status()
        return 0
    
    if not args.start_wy or not args.end_wy:
        parser.print_help()
        return 1
    
    if args.start_wy > args.end_wy:
        print("Error: start-wy must be <= end-wy")
        return 1
    
    run_batch_backfill(args.start_wy, args.end_wy, verbose=not args.quiet)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

