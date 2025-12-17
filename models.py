"""
SQLAlchemy models for the Water Risk database.

This module defines the database schema for storing SNOTEL and other
environmental monitoring data.
"""

from datetime import datetime, date
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from config import DATABASE_URL

Base = declarative_base()


class Station(Base):
    """
    SNOTEL/SCAN station metadata.
    
    Each station is uniquely identified by its triplet (station_id:state:network).
    """
    __tablename__ = "stations"
    
    id = Column(Integer, primary_key=True)
    
    # Station identification
    triplet = Column(String(50), unique=True, nullable=False, index=True)
    station_id = Column(String(20), nullable=False)
    state_code = Column(String(2), nullable=False, index=True)
    network_code = Column(String(10), nullable=False, index=True)
    
    # Station information
    name = Column(String(100), nullable=False)
    county_name = Column(String(100))
    huc = Column(String(12), index=True)  # HUC-12 watershed code
    
    # Location
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float)  # feet
    
    # Metadata
    shef_id = Column(String(20))  # SHEF identifier
    dco_code = Column(String(10))  # Data collection office
    operator = Column(String(50))
    data_timezone = Column(Float)  # UTC offset
    
    # Period of record
    begin_date = Column(Date)
    end_date = Column(Date)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    observations = relationship("DailyObservation", back_populates="station")
    
    def __repr__(self):
        return f"<Station({self.triplet}: {self.name})>"


class Element(Base):
    """
    Measurement element types (e.g., WTEQ, SNWD, PREC).
    
    Reference table for the types of measurements available.
    """
    __tablename__ = "elements"
    
    id = Column(Integer, primary_key=True)
    code = Column(String(20), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    unit = Column(String(20))  # e.g., "inches", "degrees F"
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Element({self.code}: {self.name})>"


class DailyObservation(Base):
    """
    Daily observation data from SNOTEL/SCAN stations.
    
    Stores the actual measurements along with statistical comparisons
    (median, percent of median, etc.).
    """
    __tablename__ = "daily_observations"
    
    id = Column(Integer, primary_key=True)
    
    # Foreign keys
    station_id = Column(Integer, ForeignKey("stations.id"), nullable=False)
    
    # Observation date
    observation_date = Column(Date, nullable=False, index=True)
    
    # Snow Water Equivalent (WTEQ)
    wteq_value = Column(Float)  # inches
    wteq_median = Column(Float)  # historical median for this date
    wteq_pct_median = Column(Float)  # percent of median
    wteq_delta = Column(Float)  # change from previous day
    
    # Snow Depth (SNWD)
    snwd_value = Column(Float)  # inches
    snwd_median = Column(Float)
    snwd_pct_median = Column(Float)
    snwd_delta = Column(Float)
    
    # Precipitation (PREC) - water year accumulation
    prec_value = Column(Float)  # inches
    prec_median = Column(Float)
    prec_pct_median = Column(Float)
    
    # Temperature (optional, if collected)
    tmax_value = Column(Float)  # degrees F
    tmin_value = Column(Float)
    tavg_value = Column(Float)
    
    # Data quality
    quality_flag = Column(String(10))  # e.g., 'V' for validated
    
    # Timestamps
    fetched_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    station = relationship("Station", back_populates="observations")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("station_id", "observation_date", name="uix_station_date"),
        Index("ix_obs_state_date", "observation_date"),
    )
    
    def __repr__(self):
        return f"<DailyObservation({self.station_id} @ {self.observation_date})>"


class DataFetchLog(Base):
    """
    Log of data fetch operations for auditing and debugging.
    """
    __tablename__ = "data_fetch_logs"
    
    id = Column(Integer, primary_key=True)
    
    # Fetch details
    fetch_type = Column(String(50), nullable=False)  # e.g., 'daily_snotel', 'backfill'
    state_code = Column(String(2))
    target_date = Column(Date)
    
    # Results
    stations_queried = Column(Integer)
    stations_with_data = Column(Integer)
    records_inserted = Column(Integer)
    records_updated = Column(Integer)
    
    # Status
    status = Column(String(20), nullable=False)  # 'success', 'partial', 'failed'
    error_message = Column(Text)
    
    # Timing
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)
    
    def __repr__(self):
        return f"<DataFetchLog({self.fetch_type} @ {self.started_at})>"


class BasinSummary(Base):
    """
    Pre-computed basin-level summaries for faster reporting.
    
    Aggregates station data by HUC region, state, or custom basins.
    """
    __tablename__ = "basin_summaries"
    
    id = Column(Integer, primary_key=True)
    
    # Basin identification
    basin_type = Column(String(20), nullable=False)  # 'state', 'huc2', 'huc4', 'huc6'
    basin_code = Column(String(20), nullable=False)
    basin_name = Column(String(100))
    
    # Summary date
    summary_date = Column(Date, nullable=False, index=True)
    
    # Aggregated metrics
    station_count = Column(Integer)
    stations_reporting = Column(Integer)
    
    # Snow Water Equivalent
    wteq_avg = Column(Float)
    wteq_median_avg = Column(Float)
    wteq_pct_median = Column(Float)  # Basin-wide percent of median
    wteq_min = Column(Float)
    wteq_max = Column(Float)
    
    # Snow Depth
    snwd_avg = Column(Float)
    snwd_min = Column(Float)
    snwd_max = Column(Float)
    
    # Timestamps
    computed_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint("basin_type", "basin_code", "summary_date", name="uix_basin_date"),
        Index("ix_basin_lookup", "basin_type", "basin_code", "summary_date"),
    )
    
    def __repr__(self):
        return f"<BasinSummary({self.basin_type}:{self.basin_code} @ {self.summary_date})>"


# Database initialization functions
def get_engine():
    """Create and return a database engine."""
    return create_engine(DATABASE_URL, echo=False)


def get_session():
    """Create and return a database session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    """Initialize the database schema."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    
    # Populate reference data
    session = get_session()
    
    # Add common elements if not exists
    elements = [
        {"code": "WTEQ", "name": "Snow Water Equivalent", "unit": "inches",
         "description": "The amount of water contained within the snowpack"},
        {"code": "SNWD", "name": "Snow Depth", "unit": "inches",
         "description": "The total depth of the snowpack"},
        {"code": "PREC", "name": "Precipitation Accumulation", "unit": "inches",
         "description": "Water year precipitation accumulation"},
        {"code": "TAVG", "name": "Air Temperature Average", "unit": "degrees F",
         "description": "Daily average air temperature"},
        {"code": "TMAX", "name": "Air Temperature Maximum", "unit": "degrees F",
         "description": "Daily maximum air temperature"},
        {"code": "TMIN", "name": "Air Temperature Minimum", "unit": "degrees F",
         "description": "Daily minimum air temperature"},
        {"code": "SMS", "name": "Soil Moisture", "unit": "percent",
         "description": "Soil moisture percentage"},
        {"code": "STO", "name": "Soil Temperature", "unit": "degrees F",
         "description": "Soil temperature at various depths"},
    ]
    
    for elem_data in elements:
        existing = session.query(Element).filter_by(code=elem_data["code"]).first()
        if not existing:
            elem = Element(**elem_data)
            session.add(elem)
    
    session.commit()
    session.close()
    
    print("Database initialized successfully!")


def drop_db():
    """Drop all tables (use with caution!)."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    print("All tables dropped.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--drop":
        confirm = input("Are you sure you want to drop all tables? (yes/no): ")
        if confirm.lower() == "yes":
            drop_db()
    else:
        init_db()

