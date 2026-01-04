"""
Configuration for the Water Risk data acquisition system.
Defines data sources, URLs, and preferences for spatial data downloads.

Environment variables (set in .env file):
    DATABASE_HOST - Database hostname (default: localhost)
    DATABASE_PORT - Database port (default: 5432)
    DATABASE_NAME - Database name (default: water_risk)
    DATABASE_USER - Database user (default: postgres)
    DATABASE_PASSWORD - Database password (default: empty)
    DATABASE_SSLMODE - SSL mode for connection (default: prefer)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# States of interest
STATES_OF_INTEREST = {
    "CA": {"name": "California", "fips": "06"},
    "ID": {"name": "Idaho", "fips": "16"},
    "NV": {"name": "Nevada", "fips": "32"},
    "OR": {"name": "Oregon", "fips": "41"},
    "WA": {"name": "Washington", "fips": "53"},
}

# File format preferences (in order of preference)
VECTOR_FORMAT_PREFERENCE = ["gdb", "shp", "geojson", "kml"]
RASTER_FORMAT_PREFERENCE = ["tif", "adf", "gpkg"]

# Census Bureau TIGER/Line files - 2023 vintage (most recent complete)
CENSUS_BASE_URL = "https://www2.census.gov/geo/tiger"
CENSUS_YEAR = "2023"

# Data source configurations
DATA_SOURCES = {
    # ==========================================================================
    # WATERSHED DATA - USGS Watershed Boundary Dataset
    # ==========================================================================
    "watershed_huc12": {
        "name": "USGS Watershed Boundary Dataset (WBD) - National",
        "description": "HUC-12 level watershed boundaries for the entire US",
        "source": "USGS",
        "url": "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GDB/WBD_National_GDB.zip",
        "format": "gdb",
        "category": "watershed",
        "size_estimate": "~1.5 GB",
        "priority": 1,
    },
    
    # ==========================================================================
    # POLITICAL BOUNDARIES - Census Bureau TIGER/Line
    # ==========================================================================
    "nation_boundary": {
        "name": "US National Boundary",
        "description": "United States national boundary",
        "source": "Census Bureau Cartographic Boundary",
        "url": f"{CENSUS_BASE_URL}/GENZ{CENSUS_YEAR}/shp/cb_{CENSUS_YEAR}_us_nation_5m.zip",
        "format": "shp",
        "category": "political",
        "priority": 2,
    },
    "state_boundaries": {
        "name": "US State Boundaries",
        "description": "All US state and territory boundaries",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/STATE/tl_{CENSUS_YEAR}_us_state.zip",
        "format": "shp",
        "category": "political",
        "priority": 2,
    },
    "county_boundaries": {
        "name": "US County Boundaries",
        "description": "All US county and equivalent boundaries",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/COUNTY/tl_{CENSUS_YEAR}_us_county.zip",
        "format": "shp",
        "category": "political",
        "priority": 2,
    },
    "congressional_districts_118": {
        "name": "Congressional Districts (118th Congress)",
        "description": "US Congressional district boundaries for 118th Congress",
        "source": "Census Bureau Cartographic Boundary",
        "url": f"{CENSUS_BASE_URL}/GENZ{CENSUS_YEAR}/shp/cb_{CENSUS_YEAR}_us_cd118_500k.zip",
        "format": "shp",
        "category": "political",
        "priority": 2,
    },
    
    # State-specific places (cities/towns) - need per-state downloads
    "places_california": {
        "name": "California Places (Cities/Towns)",
        "description": "Incorporated places and census designated places in California",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/PLACE/tl_{CENSUS_YEAR}_06_place.zip",
        "format": "shp",
        "category": "political",
        "state": "CA",
        "priority": 3,
    },
    "places_nevada": {
        "name": "Nevada Places (Cities/Towns)",
        "description": "Incorporated places and census designated places in Nevada",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/PLACE/tl_{CENSUS_YEAR}_32_place.zip",
        "format": "shp",
        "category": "political",
        "state": "NV",
        "priority": 3,
    },
    "places_oregon": {
        "name": "Oregon Places (Cities/Towns)",
        "description": "Incorporated places and census designated places in Oregon",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/PLACE/tl_{CENSUS_YEAR}_41_place.zip",
        "format": "shp",
        "category": "political",
        "state": "OR",
        "priority": 3,
    },
    "places_washington": {
        "name": "Washington Places (Cities/Towns)",
        "description": "Incorporated places and census designated places in Washington",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/PLACE/tl_{CENSUS_YEAR}_53_place.zip",
        "format": "shp",
        "category": "political",
        "state": "WA",
        "priority": 3,
    },
    "places_idaho": {
        "name": "Idaho Places (Cities/Towns)",
        "description": "Incorporated places and census designated places in Idaho",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/PLACE/tl_{CENSUS_YEAR}_16_place.zip",
        "format": "shp",
        "category": "political",
        "state": "ID",
        "priority": 3,
    },
    
    # ==========================================================================
    # PROTECTED AREAS - USGS PAD-US
    # ==========================================================================
    # PAD-US 4.1 State Downloads - from https://www.sciencebase.gov/catalog/item/6759abcfd34edfeb8710a004
    "padus_california": {
        "name": "PAD-US 4.1 California",
        "description": "Protected Areas Database - California (Fee, Easement, Designation, Proclamation)",
        "source": "USGS Gap Analysis Project / ScienceBase",
        "url": "https://www.sciencebase.gov/catalog/file/get/6759abcfd34edfeb8710a004?name=PADUS4_1_State_CA_GDB_KMZ.zip",
        "format": "gdb",
        "category": "protected",
        "state": "CA",
        "size_estimate": "~439 MB",
        "priority": 2,
    },
    "padus_nevada": {
        "name": "PAD-US 4.1 Nevada",
        "description": "Protected Areas Database - Nevada (Fee, Easement, Designation, Proclamation)",
        "source": "USGS Gap Analysis Project / ScienceBase",
        "url": "https://www.sciencebase.gov/catalog/file/get/6759abcfd34edfeb8710a004?name=PADUS4_1_State_NV_GDB_KMZ.zip",
        "format": "gdb",
        "category": "protected",
        "state": "NV",
        "size_estimate": "~104 MB",
        "priority": 2,
    },
    "padus_oregon": {
        "name": "PAD-US 4.1 Oregon",
        "description": "Protected Areas Database - Oregon (Fee, Easement, Designation, Proclamation)",
        "source": "USGS Gap Analysis Project / ScienceBase",
        "url": "https://www.sciencebase.gov/catalog/file/get/6759abcfd34edfeb8710a004?name=PADUS4_1_State_OR_GDB_KMZ.zip",
        "format": "gdb",
        "category": "protected",
        "state": "OR",
        "size_estimate": "~291 MB",
        "priority": 2,
    },
    "padus_washington": {
        "name": "PAD-US 4.1 Washington",
        "description": "Protected Areas Database - Washington (Fee, Easement, Designation, Proclamation)",
        "source": "USGS Gap Analysis Project / ScienceBase",
        "url": "https://www.sciencebase.gov/catalog/file/get/6759abcfd34edfeb8710a004?name=PADUS4_1_State_WA_GDB_KMZ.zip",
        "format": "gdb",
        "category": "protected",
        "state": "WA",
        "size_estimate": "~247 MB",
        "priority": 2,
    },
    "padus_idaho": {
        "name": "PAD-US 4.1 Idaho",
        "description": "Protected Areas Database - Idaho (Fee, Easement, Designation, Proclamation)",
        "source": "USGS Gap Analysis Project / ScienceBase",
        "url": "https://www.sciencebase.gov/catalog/file/get/6759abcfd34edfeb8710a004?name=PADUS4_1_State_ID_GDB_KMZ.zip",
        "format": "gdb",
        "category": "protected",
        "state": "ID",
        "size_estimate": "~200 MB",
        "priority": 2,
    },
    
    # ==========================================================================
    # SPECIAL DISTRICTS - State-specific sources
    # ==========================================================================
    # California Special Districts
    "ca_special_districts": {
        "name": "California Special Districts",
        "description": "Special district boundaries in California (water, fire, etc.)",
        "source": "California State Controller's Office",
        "url": "https://bythenumbers.sco.ca.gov/api/views/wrpa-d7d6/rows.csv?accessType=DOWNLOAD",
        "format": "csv",
        "category": "special_district",
        "state": "CA",
        "priority": 4,
        "notes": "Boundaries may need to be obtained from LAFCo or individual districts. This is fiscal data.",
        "manual_download_url": "https://gis.data.ca.gov/datasets/CALAFCO::special-district-boundaries/about",
    },
    
    # ==========================================================================
    # STATE-SPECIFIC PROTECTED AREAS (more detailed than PAD-US)
    # ==========================================================================
    "ca_protected_areas_cpad": {
        "name": "California Protected Areas Database (CPAD)",
        "description": "Detailed protected areas in California - holdings layer",
        "source": "GreenInfo Network / CALands",
        "url": "https://data.ca.gov/dataset/cpad/resource/7c56927c-5c13-4f71-a7c8-db3b7e63aa72",
        "format": "gdb",
        "category": "protected",
        "state": "CA",
        "priority": 3,
        "notes": "Download via CA Open Data Portal or visit https://www.calands.org/cpad/",
        "manual_download_url": "https://www.calands.org/cpad/",
    },
    "ca_conservation_easements": {
        "name": "California Conservation Easement Database (CCED)",
        "description": "Conservation easements in California",
        "source": "GreenInfo Network / CALands",
        "url": "https://data.ca.gov/dataset/cced",
        "format": "gdb",
        "category": "protected",
        "state": "CA",
        "priority": 3,
        "notes": "Download via CA Open Data Portal or visit https://www.calands.org/cced/",
        "manual_download_url": "https://www.calands.org/cced/",
    },
    
    # ==========================================================================
    # ADDITIONAL CENSUS BOUNDARIES
    # ==========================================================================
    # Urban Areas
    "urban_areas": {
        "name": "US Urban Areas (2020)",
        "description": "Census-defined urbanized areas and urban clusters from 2020 Census",
        "source": "Census Bureau Cartographic Boundary",
        "url": f"{CENSUS_BASE_URL}/GENZ2020/shp/cb_2020_us_ua20_500k.zip",
        "format": "shp",
        "category": "political",
        "priority": 4,
    },
    
    # American Indian/Alaska Native Areas
    "tribal_areas": {
        "name": "American Indian/Alaska Native Areas",
        "description": "Federally recognized tribal areas and reservations",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/AIANNH/tl_{CENSUS_YEAR}_us_aiannh.zip",
        "format": "shp",
        "category": "political",
        "priority": 3,
    },
    
    # State Legislative Districts
    "ca_state_senate": {
        "name": "California State Senate Districts",
        "description": "California State Senate district boundaries",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/SLDU/tl_{CENSUS_YEAR}_06_sldu.zip",
        "format": "shp",
        "category": "political",
        "state": "CA",
        "priority": 4,
    },
    "ca_state_assembly": {
        "name": "California State Assembly Districts",
        "description": "California State Assembly district boundaries",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/SLDL/tl_{CENSUS_YEAR}_06_sldl.zip",
        "format": "shp",
        "category": "political",
        "state": "CA",
        "priority": 4,
    },
    "nv_state_senate": {
        "name": "Nevada State Senate Districts",
        "description": "Nevada State Senate district boundaries",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/SLDU/tl_{CENSUS_YEAR}_32_sldu.zip",
        "format": "shp",
        "category": "political",
        "state": "NV",
        "priority": 4,
    },
    "nv_state_assembly": {
        "name": "Nevada State Assembly Districts",
        "description": "Nevada State Assembly district boundaries",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/SLDL/tl_{CENSUS_YEAR}_32_sldl.zip",
        "format": "shp",
        "category": "political",
        "state": "NV",
        "priority": 4,
    },
    "or_state_senate": {
        "name": "Oregon State Senate Districts",
        "description": "Oregon State Senate district boundaries",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/SLDU/tl_{CENSUS_YEAR}_41_sldu.zip",
        "format": "shp",
        "category": "political",
        "state": "OR",
        "priority": 4,
    },
    "or_state_house": {
        "name": "Oregon State House Districts",
        "description": "Oregon State House district boundaries",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/SLDL/tl_{CENSUS_YEAR}_41_sldl.zip",
        "format": "shp",
        "category": "political",
        "state": "OR",
        "priority": 4,
    },
    "wa_state_senate": {
        "name": "Washington State Senate Districts",
        "description": "Washington State Legislative district boundaries (Senate)",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/SLDU/tl_{CENSUS_YEAR}_53_sldu.zip",
        "format": "shp",
        "category": "political",
        "state": "WA",
        "priority": 4,
    },
    "wa_state_house": {
        "name": "Washington State House Districts",
        "description": "Washington State Legislative district boundaries (House)",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/SLDL/tl_{CENSUS_YEAR}_53_sldl.zip",
        "format": "shp",
        "category": "political",
        "state": "WA",
        "priority": 4,
    },
    "id_state_senate": {
        "name": "Idaho State Senate Districts",
        "description": "Idaho State Senate district boundaries",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/SLDU/tl_{CENSUS_YEAR}_16_sldu.zip",
        "format": "shp",
        "category": "political",
        "state": "ID",
        "priority": 4,
    },
    "id_state_house": {
        "name": "Idaho State House Districts",
        "description": "Idaho State House district boundaries",
        "source": "Census Bureau TIGER/Line",
        "url": f"{CENSUS_BASE_URL}/TIGER{CENSUS_YEAR}/SLDL/tl_{CENSUS_YEAR}_16_sldl.zip",
        "format": "shp",
        "category": "political",
        "state": "ID",
        "priority": 4,
    },
}

# Download settings
DOWNLOAD_SETTINGS = {
    "chunk_size": 8192,
    "timeout": 300,  # 5 minutes for large files
    "max_retries": 3,
    "retry_delay": 5,  # seconds
    "verify_ssl": True,
}

# Database configuration (loads from environment variables)
DATABASE_CONFIG = {
    "host": os.getenv("DATABASE_HOST", "localhost"),
    "port": int(os.getenv("DATABASE_PORT", "5432")),
    "database": os.getenv("DATABASE_NAME", "water_risk"),
    "user": os.getenv("DATABASE_USER", "postgres"),
    "password": os.getenv("DATABASE_PASSWORD", ""),
    "sslmode": os.getenv("DATABASE_SSLMODE", "prefer"),
}

# Build connection URL
_password_part = f":{DATABASE_CONFIG['password']}" if DATABASE_CONFIG['password'] else ""
_ssl_part = f"?sslmode={DATABASE_CONFIG['sslmode']}" if DATABASE_CONFIG['sslmode'] != "prefer" else ""

DATABASE_URL = (
    f"postgresql://{DATABASE_CONFIG['user']}{_password_part}"
    f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}"
    f"/{DATABASE_CONFIG['database']}{_ssl_part}"
)

