#!/usr/bin/env python3
"""
Database Migration Script

Migrates SNOTEL data from local PostgreSQL to remote GeoServer database.

Usage:
    python migrate_to_geoserver.py --password YOUR_PASSWORD
    
    # Test connection first
    python migrate_to_geoserver.py --password YOUR_PASSWORD --test
    
    # Migrate specific tables
    python migrate_to_geoserver.py --password YOUR_PASSWORD --tables stations,daily_observations
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# Remote database configuration
REMOTE_CONFIG = {
    "host": "ocd-20251019-do-user-9716067-0.l.db.ondigitalocean.com",
    "port": "25060",
    "user": "doadmin",
    "sslmode": "require",
}

# Local database configuration
LOCAL_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "water_risk",
    "user": "kylebirchard",
}

# Tables to migrate (in order due to foreign keys)
TABLES_TO_MIGRATE = [
    "elements",
    "stations", 
    "daily_observations",
    "basin_summaries",
    "data_fetch_logs",
]


def test_connection(password: str, database: str = "defaultdb") -> bool:
    """Test connection to remote database."""
    print(f"Testing connection to {REMOTE_CONFIG['host']}...")
    
    env = os.environ.copy()
    env["PGPASSWORD"] = password
    
    cmd = [
        "psql",
        "-h", REMOTE_CONFIG["host"],
        "-p", REMOTE_CONFIG["port"],
        "-U", REMOTE_CONFIG["user"],
        "-d", database,
        f"sslmode={REMOTE_CONFIG['sslmode']}",
        "-c", "SELECT version();",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode == 0:
        print("✓ Connection successful!")
        print(result.stdout)
        return True
    else:
        print("✗ Connection failed!")
        print(result.stderr)
        return False


def create_remote_database(password: str, database: str = "water_risk") -> bool:
    """Create the database on the remote server."""
    print(f"Creating database '{database}' on remote server...")
    
    env = os.environ.copy()
    env["PGPASSWORD"] = password
    
    # Connect to default database to create new one
    cmd = [
        "psql",
        "-h", REMOTE_CONFIG["host"],
        "-p", REMOTE_CONFIG["port"],
        "-U", REMOTE_CONFIG["user"],
        "-d", "defaultdb",
        f"sslmode={REMOTE_CONFIG['sslmode']}",
        "-c", f"CREATE DATABASE {database};",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode == 0 or "already exists" in result.stderr:
        print(f"✓ Database '{database}' ready")
        return True
    else:
        print(f"Note: {result.stderr.strip()}")
        return True  # May already exist


def enable_postgis(password: str, database: str = "water_risk") -> bool:
    """Enable PostGIS extension on remote database."""
    print("Enabling PostGIS extension...")
    
    env = os.environ.copy()
    env["PGPASSWORD"] = password
    
    cmd = [
        "psql",
        "-h", REMOTE_CONFIG["host"],
        "-p", REMOTE_CONFIG["port"],
        "-U", REMOTE_CONFIG["user"],
        "-d", database,
        f"sslmode={REMOTE_CONFIG['sslmode']}",
        "-c", "CREATE EXTENSION IF NOT EXISTS postgis;",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode == 0:
        print("✓ PostGIS enabled")
        return True
    else:
        print(f"Warning: {result.stderr.strip()}")
        return False


def export_schema(output_file: str = "schema.sql") -> bool:
    """Export schema from local database."""
    print("Exporting schema from local database...")
    
    cmd = [
        "pg_dump",
        "-h", LOCAL_CONFIG["host"],
        "-p", LOCAL_CONFIG["port"],
        "-U", LOCAL_CONFIG["user"],
        "-d", LOCAL_CONFIG["database"],
        "--schema-only",
        "--no-owner",
        "--no-privileges",
        "-f", output_file,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Schema exported to {output_file}")
        return True
    else:
        print(f"✗ Export failed: {result.stderr}")
        return False


def export_table_data(table: str, output_file: str) -> bool:
    """Export data from a specific table."""
    print(f"  Exporting {table}...")
    
    cmd = [
        "pg_dump",
        "-h", LOCAL_CONFIG["host"],
        "-p", LOCAL_CONFIG["port"],
        "-U", LOCAL_CONFIG["user"],
        "-d", LOCAL_CONFIG["database"],
        "--data-only",
        "--table", table,
        "-f", output_file,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        size = Path(output_file).stat().st_size / (1024 * 1024)
        print(f"    ✓ Exported ({size:.1f} MB)")
        return True
    else:
        print(f"    ✗ Failed: {result.stderr}")
        return False


def import_to_remote(sql_file: str, password: str, database: str = "water_risk") -> bool:
    """Import SQL file to remote database."""
    print(f"  Importing {sql_file}...")
    
    env = os.environ.copy()
    env["PGPASSWORD"] = password
    
    cmd = [
        "psql",
        "-h", REMOTE_CONFIG["host"],
        "-p", REMOTE_CONFIG["port"],
        "-U", REMOTE_CONFIG["user"],
        "-d", database,
        f"sslmode={REMOTE_CONFIG['sslmode']}",
        "-f", sql_file,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode == 0:
        print(f"    ✓ Imported")
        return True
    else:
        # Check for common non-fatal errors
        if "already exists" in result.stderr:
            print(f"    ✓ Already exists (skipped)")
            return True
        print(f"    ✗ Failed: {result.stderr[:200]}")
        return False


def run_remote_sql(sql: str, password: str, database: str = "water_risk") -> bool:
    """Run SQL command on remote database."""
    env = os.environ.copy()
    env["PGPASSWORD"] = password
    
    cmd = [
        "psql",
        "-h", REMOTE_CONFIG["host"],
        "-p", REMOTE_CONFIG["port"],
        "-U", REMOTE_CONFIG["user"],
        "-d", database,
        f"sslmode={REMOTE_CONFIG['sslmode']}",
        "-c", sql,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.returncode == 0


def migrate_full(password: str, database: str = "water_risk", tables: list = None):
    """Run full migration."""
    if tables is None:
        tables = TABLES_TO_MIGRATE
    
    export_dir = Path("migration_export")
    export_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("SNOTEL Database Migration")
    print("=" * 60)
    print(f"From: {LOCAL_CONFIG['host']}:{LOCAL_CONFIG['port']}/{LOCAL_CONFIG['database']}")
    print(f"To:   {REMOTE_CONFIG['host']}:{REMOTE_CONFIG['port']}/{database}")
    print(f"Tables: {', '.join(tables)}")
    print("=" * 60)
    print()
    
    # Step 1: Test connection
    if not test_connection(password):
        return False
    print()
    
    # Step 2: Create database
    create_remote_database(password, database)
    print()
    
    # Step 3: Enable PostGIS
    enable_postgis(password, database)
    print()
    
    # Step 4: Export and import schema
    schema_file = str(export_dir / "schema.sql")
    if export_schema(schema_file):
        import_to_remote(schema_file, password, database)
    print()
    
    # Step 5: Export and import GeoServer setup
    print("Importing GeoServer views...")
    geoserver_sql = Path("setup_geoserver.sql")
    if geoserver_sql.exists():
        import_to_remote(str(geoserver_sql), password, database)
    print()
    
    # Step 6: Export and import data table by table
    print("Migrating table data...")
    for table in tables:
        data_file = str(export_dir / f"{table}_data.sql")
        if export_table_data(table, data_file):
            import_to_remote(data_file, password, database)
    print()
    
    # Step 7: Refresh materialized view
    print("Refreshing materialized views...")
    run_remote_sql("SELECT refresh_snotel_latest();", password, database)
    print()
    
    # Step 8: Verify
    print("Verifying migration...")
    env = os.environ.copy()
    env["PGPASSWORD"] = password
    
    verify_cmd = [
        "psql",
        "-h", REMOTE_CONFIG["host"],
        "-p", REMOTE_CONFIG["port"],
        "-U", REMOTE_CONFIG["user"],
        "-d", database,
        f"sslmode={REMOTE_CONFIG['sslmode']}",
        "-c", """
            SELECT 'stations' as table_name, COUNT(*) as rows FROM stations
            UNION ALL
            SELECT 'daily_observations', COUNT(*) FROM daily_observations
            UNION ALL
            SELECT 'elements', COUNT(*) FROM elements;
        """,
    ]
    
    result = subprocess.run(verify_cmd, capture_output=True, text=True, env=env)
    print(result.stdout)
    
    print("=" * 60)
    print("Migration complete!")
    print("=" * 60)
    print()
    print("GeoServer connection details:")
    print(f"  Host: {REMOTE_CONFIG['host']}")
    print(f"  Port: {REMOTE_CONFIG['port']}")
    print(f"  Database: {database}")
    print(f"  User: {REMOTE_CONFIG['user']}")
    print(f"  SSL Mode: {REMOTE_CONFIG['sslmode']}")
    print()
    print("Available WFS layers:")
    print("  - vw_snotel_stations (static stations)")
    print("  - mv_snotel_latest (latest observations)")
    print("  - get_snotel_by_date(date) (historical data)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Migrate SNOTEL data to GeoServer")
    
    parser.add_argument(
        "--password",
        type=str,
        required=True,
        help="Password for remote database"
    )
    parser.add_argument(
        "--database",
        type=str,
        default="water_risk",
        help="Remote database name (default: water_risk)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Only test the connection"
    )
    parser.add_argument(
        "--tables",
        type=str,
        help="Comma-separated list of tables to migrate"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_connection(args.password)
        return 0
    
    tables = None
    if args.tables:
        tables = [t.strip() for t in args.tables.split(",")]
    
    migrate_full(args.password, args.database, tables)
    return 0


if __name__ == "__main__":
    sys.exit(main())

