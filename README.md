# Water Risk Application

Environmental data acquisition and analysis system for water risk assessment across California, Idaho, Nevada, Oregon, and Washington. Integrates SNOTEL snowpack data with geospatial boundaries for watershed-scale analysis.

## Features

- **SNOTEL Data Collection**: Automated daily collection of snow water equivalent, snow depth, precipitation, and temperature data from 339 stations
- **Historical Database**: 30 years of daily observations (1995-present) stored in PostgreSQL
- **Geospatial Integration**: HUC-12 watersheds, political boundaries, protected areas
- **Visualization**: Interactive maps with Basin Index color ramps

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize the database
createdb water_risk
python models.py

# Sync station metadata
python collect_snotel_data.py --sync-stations

# Collect yesterday's data
python daily_snotel_update.py

# Check data status
python backfill_water_years.py --status

# Create snowpack visualization
python visualize_snowpack.py --state OR
```

## Project Structure

```
water_risk/
├── config.py                    # Configuration and data sources
├── models.py                    # Database schema (SQLAlchemy)
├── nwcc_api.py                  # NWCC AWDB REST API client
├── collect_snotel_data.py       # Optimized data collection
├── daily_snotel_update.py       # Daily cron job script
├── backfill_water_years.py      # Historical backfill by water year
├── visualize_snowpack.py        # Interactive map generation
├── download_spatial_data.py     # Geospatial data downloads
├── process_spatial_data.py      # Geospatial processing utilities
├── requirements.txt             # Python dependencies
├── data/
│   ├── raw/                     # Downloaded geospatial datasets
│   └── processed/               # Analysis-ready outputs
└── logs/                        # Daily update logs
```

## Database Schema

### Tables

| Table | Description |
|-------|-------------|
| `stations` | SNOTEL station metadata (339 stations) |
| `elements` | Measurement types (WTEQ, SNWD, PREC, etc.) |
| `daily_observations` | Daily measurements (~2.5M records) |
| `basin_summaries` | Pre-computed basin aggregations |
| `data_fetch_logs` | Audit trail for data collection |

### Key Fields (daily_observations)

| Field | Description |
|-------|-------------|
| `wteq_value` | Snow Water Equivalent (inches) |
| `wteq_median` | Historical median for comparison |
| `wteq_pct_median` | Basin Index (% of median) |
| `snwd_value` | Snow Depth (inches) |
| `prec_value` | Precipitation accumulation |
| `tmax/tmin/tavg` | Temperature (°F) |

## Data Collection

### Daily Updates (Cron Job)

```bash
# Run daily at 6 AM
0 6 * * * cd /path/to/water_risk && python daily_snotel_update.py >> logs/daily_update.log 2>&1
```

### Manual Collection

```bash
# Fetch specific date
python daily_snotel_update.py --date 2025-12-15

# Catch up after downtime
python daily_snotel_update.py --catch-up

# Fetch last 7 days
python daily_snotel_update.py --days 7
```

### Historical Backfill

```bash
# Backfill water year range (WY = Oct 1 - Sep 30)
python backfill_water_years.py --start-wy 1996 --end-wy 2024

# Check status
python backfill_water_years.py --status
```

## NWCC API Client

```python
from nwcc_api import NWCCClient

client = NWCCClient()

# Get station metadata
stations = client.get_stations(state="OR", network="SNTL")

# Get data for stations
data = client.get_data(
    station_triplets=["302:OR:SNTL", "304:OR:SNTL"],
    elements=["WTEQ", "SNWD"],
    begin_date="2025-01-01",
    end_date="2025-01-31",
    include_median=True,
)
```

## Visualization

```bash
# Generate interactive snowpack map
python visualize_snowpack.py --state OR

# Output: data/processed/snowpack_map_or.html
```

Features:
- HUC-12 watershed boundaries
- SNOTEL stations as colored markers
- Red-Yellow-Green color ramp (0-100% of median)
- Interactive popups with station details

## Geospatial Data

### Automatically Downloaded

| Dataset | Source | Format |
|---------|--------|--------|
| Watershed (HUC-12) | USGS WBD | GDB |
| State/County Boundaries | Census TIGER | SHP |
| Congressional Districts | Census Bureau | SHP |
| Places (Cities/Towns) | Census TIGER | SHP |
| State Legislative Districts | Census TIGER | SHP |
| Tribal Areas | Census TIGER | SHP |
| Protected Areas (PAD-US) | USGS GAP | GDB |

### Download Commands

```bash
# List available datasets
python download_spatial_data.py --list

# Download all datasets
python download_spatial_data.py --all

# Download specific category
python download_spatial_data.py --category political
```

## States of Interest

| State | FIPS | SNOTEL Stations | Earliest Data |
|-------|------|-----------------|---------------|
| California | 06 | 36 | 1977-10-01 |
| Idaho | 16 | 86 | 1978-09-30 |
| Nevada | 32 | 57 | 1976-03-01 |
| Oregon | 41 | 82 | 1978-10-01 |
| Washington | 53 | 78 | 1978-10-01 |

## Water Year Convention

A **Water Year (WY)** runs from October 1 to September 30:
- WY 2025 = October 1, 2024 through September 30, 2025

This aligns with the hydrologic cycle in the western US where precipitation primarily falls as snow in winter months.

## Dependencies

### Core
- Python 3.10+
- PostgreSQL 14+
- SQLAlchemy 2.0+

### Data Processing
- geopandas, shapely, pyproj
- pandas, numpy
- aiohttp (async API calls)

### Visualization
- folium (interactive maps)
- matplotlib

See `requirements.txt` for complete list.

## Configuration

Edit `config.py` to modify:
- `STATES_OF_INTEREST`: States to include
- `DATABASE_CONFIG`: PostgreSQL connection
- `DATA_SOURCES`: Geospatial dataset URLs

## License

Data sources have individual licenses:
- **USGS Data**: Public domain
- **Census Bureau**: Public domain  
- **NWCC/SNOTEL**: Public domain

## Acknowledgments

- USDA Natural Resources Conservation Service (NRCS) for SNOTEL data
- USGS for Watershed Boundary Dataset and PAD-US
- US Census Bureau for TIGER/Line boundaries
