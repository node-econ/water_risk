#!/usr/bin/env python3
"""
Daily SNOTEL Data Update Script

Fetches the latest SNOTEL data and stores it in PostgreSQL.
Designed to be run as a daily cron job.

Usage:
    # Fetch yesterday's data (default)
    python daily_snotel_update.py
    
    # Fetch specific date
    python daily_snotel_update.py --date 2025-12-15
    
    # Fetch last N days (catch up after downtime)
    python daily_snotel_update.py --days 7
    
    # Dry run (show what would be fetched)
    python daily_snotel_update.py --dry-run

Cron Setup:
    # Run daily at 6 AM (data is typically available by then)
    0 6 * * * cd /Users/kylebirchard/Projects/water_risk && python daily_snotel_update.py >> logs/daily_update.log 2>&1
"""

import argparse
import sys
import logging
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

# Setup logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "daily_update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import after logging setup
from sqlalchemy import text
from models import get_session
from collect_snotel_data import collect_daily_data, fetch_date_range_async, process_api_data, ELEMENTS
from config import STATES_OF_INTEREST


def get_latest_date() -> date:
    """Get the most recent date in the database."""
    session = get_session()
    result = session.execute(text("SELECT MAX(observation_date) FROM daily_observations")).scalar()
    session.close()
    return result


def get_missing_dates(start_date: date, end_date: date) -> list[date]:
    """Find dates in range that have no or incomplete data."""
    session = get_session()
    
    # Get dates with data and their observation counts
    result = session.execute(text("""
        SELECT observation_date, COUNT(*) as obs_count
        FROM daily_observations
        WHERE observation_date BETWEEN :start AND :end
        GROUP BY observation_date
    """), {"start": start_date, "end": end_date}).fetchall()
    
    session.close()
    
    # Create set of dates with sufficient data (at least 200 observations)
    dates_with_data = {row[0] for row in result if row[1] >= 200}
    
    # Find missing dates
    missing = []
    current = start_date
    while current <= end_date:
        if current not in dates_with_data:
            missing.append(current)
        current += timedelta(days=1)
    
    return missing


def update_single_date(target_date: date) -> dict:
    """Fetch and store data for a single date."""
    logger.info(f"Fetching data for {target_date}")
    
    stats = collect_daily_data(target_date)
    
    logger.info(
        f"  Stations: {stats['stations_with_data']}/{stats['stations_queried']}, "
        f"Records: {stats['records_inserted']}"
    )
    
    return stats


def update_date_range(start_date: date, end_date: date) -> dict:
    """Fetch and store data for a date range."""
    import asyncio
    from models import Station
    
    session = get_session()
    
    # Get all active stations
    states = list(STATES_OF_INTEREST.keys())
    db_stations = session.query(Station).filter(
        Station.state_code.in_(states),
        Station.network_code == "SNTL",
        Station.is_active == True,
    ).all()
    
    stations_dict = {s.triplet: s for s in db_stations}
    triplets = list(stations_dict.keys())
    
    logger.info(f"Fetching {start_date} to {end_date} for {len(triplets)} stations")
    
    # Fetch data
    api_data = asyncio.run(fetch_date_range_async(triplets, start_date, end_date))
    
    # Process and insert
    stats = process_api_data(api_data, stations_dict, session)
    
    session.close()
    
    logger.info(f"  Inserted {stats['observations']} observations for {len(stats['stations'])} stations")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Daily SNOTEL Data Update",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--date",
        type=str,
        help="Specific date to fetch (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of days to fetch (default: 1 = yesterday)"
    )
    parser.add_argument(
        "--catch-up",
        action="store_true",
        help="Fetch all missing dates since last update"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without actually fetching"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("SNOTEL Daily Update")
    logger.info("=" * 60)
    
    # Determine date(s) to fetch
    yesterday = date.today() - timedelta(days=1)
    
    if args.date:
        # Specific date
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        dates_to_fetch = [target_date]
    elif args.catch_up:
        # All missing dates since last update
        latest = get_latest_date()
        if latest:
            start = latest + timedelta(days=1)
            dates_to_fetch = get_missing_dates(start, yesterday)
        else:
            logger.error("No data in database. Run initial backfill first.")
            return 1
    elif args.days > 1:
        # Last N days
        start = yesterday - timedelta(days=args.days - 1)
        dates_to_fetch = [start + timedelta(days=i) for i in range(args.days)]
    else:
        # Just yesterday (default)
        dates_to_fetch = [yesterday]
    
    if not dates_to_fetch:
        logger.info("No dates to fetch - database is up to date!")
        return 0
    
    logger.info(f"Dates to fetch: {len(dates_to_fetch)}")
    if len(dates_to_fetch) <= 10:
        for d in dates_to_fetch:
            logger.info(f"  - {d}")
    else:
        logger.info(f"  - {dates_to_fetch[0]} to {dates_to_fetch[-1]}")
    
    if args.dry_run:
        logger.info("Dry run - no data fetched")
        return 0
    
    # Fetch data
    total_records = 0
    
    if len(dates_to_fetch) == 1:
        # Single date
        stats = update_single_date(dates_to_fetch[0])
        total_records = stats["records_inserted"]
    elif len(dates_to_fetch) <= 30:
        # Small range - fetch as one batch
        stats = update_date_range(min(dates_to_fetch), max(dates_to_fetch))
        total_records = stats["observations"]
    else:
        # Large range - fetch in chunks
        for i in range(0, len(dates_to_fetch), 30):
            chunk = dates_to_fetch[i:i + 30]
            stats = update_date_range(min(chunk), max(chunk))
            total_records += stats["observations"]
    
    logger.info("=" * 60)
    logger.info(f"Update complete! {total_records} records added.")
    logger.info("=" * 60)
    
    # Show current status
    latest = get_latest_date()
    logger.info(f"Database now current through: {latest}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

