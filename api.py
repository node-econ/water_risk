#!/usr/bin/env python3
"""
Water Risk API

FastAPI backend for serving SNOTEL data to the frontend.

Usage:
    uvicorn api:app --reload --port 8000
    
    # Or with Python
    python api.py
"""

import os
from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import text

from config import DATABASE_URL, STATES_OF_INTEREST
from models import get_session

app = FastAPI(
    title="Water Risk API",
    description="SNOTEL snowpack data API",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class StationResponse(BaseModel):
    triplet: str
    station_name: str
    state_code: str
    county_name: Optional[str]
    huc: Optional[str]
    elevation: Optional[float]
    latitude: float
    longitude: float
    observation_date: Optional[date]
    wteq_value: Optional[float]
    wteq_median: Optional[float]
    basin_index: Optional[float]
    snow_depth: Optional[float]
    precip: Optional[float]
    temp_max: Optional[float]
    temp_min: Optional[float]


class DateRangeResponse(BaseModel):
    earliest: date
    latest: date
    total_dates: int


class StatsResponse(BaseModel):
    date: date
    state_code: Optional[str]
    station_count: int
    stations_reporting: int
    avg_basin_index: Optional[float]
    min_basin_index: Optional[float]
    max_basin_index: Optional[float]
    avg_wteq: Optional[float]
    avg_snow_depth: Optional[float]


# Routes
@app.get("/")
async def root():
    """Serve the frontend."""
    return FileResponse("frontend/index.html")


@app.get("/api/dates")
async def get_available_dates() -> DateRangeResponse:
    """Get the range of available dates in the database."""
    session = get_session()
    
    result = session.execute(text("""
        SELECT 
            MIN(observation_date) as earliest,
            MAX(observation_date) as latest,
            COUNT(DISTINCT observation_date) as total_dates
        FROM daily_observations
    """)).fetchone()
    
    session.close()
    
    return DateRangeResponse(
        earliest=result[0],
        latest=result[1],
        total_dates=result[2]
    )


@app.get("/api/observations")
async def get_observations(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    state: Optional[str] = Query(None, description="State code (CA, ID, NV, OR, WA)"),
) -> list[dict]:
    """
    Get SNOTEL observations for a specific date.
    
    Returns GeoJSON-like features for mapping.
    """
    session = get_session()
    
    # Default to latest date if not specified
    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        result = session.execute(text("SELECT MAX(observation_date) FROM daily_observations")).scalar()
        target_date = result
    
    # Build query
    query = """
        SELECT 
            s.triplet,
            s.name as station_name,
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
        LEFT JOIN daily_observations d ON s.id = d.station_id AND d.observation_date = :target_date
        WHERE s.is_active = true
          AND s.latitude IS NOT NULL
          AND s.longitude IS NOT NULL
    """
    
    params = {"target_date": target_date}
    
    if state and state.upper() != "ALL":
        query += " AND s.state_code = :state"
        params["state"] = state.upper()
    
    query += " ORDER BY s.state_code, s.name"
    
    result = session.execute(text(query), params).fetchall()
    session.close()
    
    # Convert to GeoJSON-like format
    features = []
    for row in result:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row.longitude, row.latitude]
            },
            "properties": {
                "triplet": row.triplet,
                "station_name": row.station_name,
                "state_code": row.state_code,
                "county_name": row.county_name,
                "huc": row.huc,
                "elevation": row.elevation,
                "latitude": row.latitude,
                "longitude": row.longitude,
                "observation_date": str(row.observation_date) if row.observation_date else None,
                "wteq_value": row.wteq_value,
                "wteq_median": row.wteq_median,
                "basin_index": row.basin_index,
                "snow_depth": row.snow_depth,
                "precip": row.precip,
                "temp_max": row.temp_max,
                "temp_min": row.temp_min,
            }
        })
    
    return features


@app.get("/api/stations")
async def get_stations(
    state: Optional[str] = Query(None, description="State code"),
) -> list[dict]:
    """Get all SNOTEL station metadata."""
    session = get_session()
    
    query = """
        SELECT 
            triplet,
            name as station_name,
            state_code,
            county_name,
            huc,
            elevation,
            latitude,
            longitude,
            begin_date,
            is_active
        FROM stations
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    """
    
    params = {}
    if state:
        query += " AND state_code = :state"
        params["state"] = state.upper()
    
    query += " ORDER BY state_code, name"
    
    result = session.execute(text(query), params).fetchall()
    session.close()
    
    return [dict(row._mapping) for row in result]


@app.get("/api/stats")
async def get_stats(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    state: Optional[str] = Query(None, description="State code"),
) -> StatsResponse:
    """Get aggregate statistics for a date."""
    session = get_session()
    
    if date:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    else:
        target_date = session.execute(text("SELECT MAX(observation_date) FROM daily_observations")).scalar()
    
    query = """
        SELECT 
            :target_date as date,
            :state_code as state_code,
            COUNT(DISTINCT s.id) as station_count,
            COUNT(d.wteq_value) as stations_reporting,
            AVG(d.wteq_pct_median) as avg_basin_index,
            MIN(d.wteq_pct_median) as min_basin_index,
            MAX(d.wteq_pct_median) as max_basin_index,
            AVG(d.wteq_value) as avg_wteq,
            AVG(d.snwd_value) as avg_snow_depth
        FROM stations s
        LEFT JOIN daily_observations d ON s.id = d.station_id AND d.observation_date = :target_date
        WHERE s.is_active = true
    """
    
    params = {"target_date": target_date, "state_code": state}
    
    if state:
        query += " AND s.state_code = :state_filter"
        params["state_filter"] = state.upper()
    
    result = session.execute(text(query), params).fetchone()
    session.close()
    
    return StatsResponse(
        date=target_date,
        state_code=state,
        station_count=result.station_count,
        stations_reporting=result.stations_reporting,
        avg_basin_index=round(result.avg_basin_index, 1) if result.avg_basin_index else None,
        min_basin_index=round(result.min_basin_index, 1) if result.min_basin_index else None,
        max_basin_index=round(result.max_basin_index, 1) if result.max_basin_index else None,
        avg_wteq=round(result.avg_wteq, 2) if result.avg_wteq else None,
        avg_snow_depth=round(result.avg_snow_depth, 1) if result.avg_snow_depth else None,
    )


@app.get("/api/timeseries/{triplet}")
async def get_timeseries(
    triplet: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
) -> list[dict]:
    """Get time series data for a specific station."""
    session = get_session()
    
    # Default to current water year
    if not end_date:
        end = session.execute(text("SELECT MAX(observation_date) FROM daily_observations")).scalar()
    else:
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    if not start_date:
        # Default to start of water year
        if end.month >= 10:
            start = date(end.year, 10, 1)
        else:
            start = date(end.year - 1, 10, 1)
    else:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
    
    query = """
        SELECT 
            d.observation_date,
            d.wteq_value,
            d.wteq_median,
            d.wteq_pct_median as basin_index,
            d.snwd_value as snow_depth,
            d.prec_value as precip
        FROM daily_observations d
        JOIN stations s ON d.station_id = s.id
        WHERE s.triplet = :triplet
          AND d.observation_date BETWEEN :start_date AND :end_date
        ORDER BY d.observation_date
    """
    
    result = session.execute(text(query), {
        "triplet": triplet,
        "start_date": start,
        "end_date": end,
    }).fetchall()
    
    session.close()
    
    return [
        {
            "date": str(row.observation_date),
            "wteq_value": row.wteq_value,
            "wteq_median": row.wteq_median,
            "basin_index": row.basin_index,
            "snow_depth": row.snow_depth,
            "precip": row.precip,
        }
        for row in result
    ]


# Serve static files (frontend)
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

