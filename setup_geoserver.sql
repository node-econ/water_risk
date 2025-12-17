-- ============================================================================
-- GeoServer WFS Setup for SNOTEL Data
-- ============================================================================
-- This script creates PostGIS-enabled views for serving via GeoServer WFS.
--
-- Prerequisites:
--   1. PostgreSQL with PostGIS extension
--   2. water_risk database with populated stations and daily_observations tables
--   3. GeoServer with PostGIS datastore connection
--
-- GeoServer Configuration:
--   1. Create a new PostGIS Data Store pointing to water_risk database
--   2. Publish these layers:
--      - vw_snotel_stations (static station layer)
--      - mv_snotel_latest (latest observations - refresh daily)
--      - SQL View with parameter for historical dates
-- ============================================================================

-- Enable PostGIS if not already enabled
CREATE EXTENSION IF NOT EXISTS postgis;

-- ============================================================================
-- STATION GEOMETRY
-- ============================================================================
-- Add geometry column to stations table if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'stations' AND column_name = 'geom'
    ) THEN
        ALTER TABLE stations ADD COLUMN geom geometry(Point, 4326);
    END IF;
END $$;

-- Populate geometry from lat/lon
UPDATE stations 
SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326) 
WHERE geom IS NULL 
  AND latitude IS NOT NULL 
  AND longitude IS NOT NULL;

-- Create spatial index
CREATE INDEX IF NOT EXISTS idx_stations_geom ON stations USING GIST (geom);

-- ============================================================================
-- VIEW 1: Station Metadata (Static Layer)
-- ============================================================================
-- Use this for a base layer showing all station locations
DROP VIEW IF EXISTS vw_snotel_stations CASCADE;

CREATE VIEW vw_snotel_stations AS
SELECT 
    s.id,
    s.triplet,
    s.station_id,
    s.state_code,
    s.name,
    s.county_name,
    s.huc,
    s.elevation,
    s.latitude,
    s.longitude,
    s.begin_date,
    s.end_date,
    s.is_active,
    s.geom
FROM stations s
WHERE s.geom IS NOT NULL;

COMMENT ON VIEW vw_snotel_stations IS 'SNOTEL station metadata with geometry for WFS';

-- ============================================================================
-- VIEW 2: Latest Observations (Dynamic Layer)
-- ============================================================================
-- Shows the most recent data for each station
DROP VIEW IF EXISTS vw_snotel_latest CASCADE;

CREATE VIEW vw_snotel_latest AS
SELECT 
    s.id as station_id,
    s.triplet,
    s.name as station_name,
    s.state_code,
    s.county_name,
    s.huc,
    s.elevation,
    d.observation_date,
    d.wteq_value,
    d.wteq_median,
    d.wteq_pct_median as basin_index,
    d.snwd_value as snow_depth,
    d.snwd_median,
    d.prec_value as precip,
    d.prec_median,
    d.tmax_value as temp_max,
    d.tmin_value as temp_min,
    d.tavg_value as temp_avg,
    -- Color classification for styling
    CASE 
        WHEN d.wteq_pct_median IS NULL THEN 'no-data'
        WHEN d.wteq_pct_median < 25 THEN 'critical'
        WHEN d.wteq_pct_median < 50 THEN 'low'
        WHEN d.wteq_pct_median < 75 THEN 'below-normal'
        WHEN d.wteq_pct_median < 90 THEN 'near-normal'
        WHEN d.wteq_pct_median < 110 THEN 'normal'
        WHEN d.wteq_pct_median < 150 THEN 'above-normal'
        ELSE 'high'
    END as condition_class,
    s.geom
FROM stations s
JOIN daily_observations d ON s.id = d.station_id
WHERE d.observation_date = (SELECT MAX(observation_date) FROM daily_observations)
  AND s.geom IS NOT NULL;

COMMENT ON VIEW vw_snotel_latest IS 'Latest SNOTEL observations for WFS - updates automatically';

-- ============================================================================
-- MATERIALIZED VIEW: Latest Observations (Performance)
-- ============================================================================
-- For better performance, use materialized view and refresh periodically
DROP MATERIALIZED VIEW IF EXISTS mv_snotel_latest CASCADE;

CREATE MATERIALIZED VIEW mv_snotel_latest AS
SELECT * FROM vw_snotel_latest;

CREATE INDEX idx_mv_snotel_latest_geom ON mv_snotel_latest USING GIST (geom);
CREATE INDEX idx_mv_snotel_latest_state ON mv_snotel_latest (state_code);
CREATE INDEX idx_mv_snotel_latest_date ON mv_snotel_latest (observation_date);

COMMENT ON MATERIALIZED VIEW mv_snotel_latest IS 'Materialized latest observations - refresh with refresh_snotel_latest()';

-- Function to refresh the materialized view (call from cron or after daily update)
CREATE OR REPLACE FUNCTION refresh_snotel_latest()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW mv_snotel_latest;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEW 3: State Summary (Non-spatial)
-- ============================================================================
DROP VIEW IF EXISTS vw_snotel_state_summary CASCADE;

CREATE VIEW vw_snotel_state_summary AS
SELECT 
    s.state_code,
    d.observation_date,
    COUNT(*) as station_count,
    COUNT(d.wteq_value) as stations_reporting,
    ROUND(AVG(d.wteq_value)::numeric, 2) as avg_wteq,
    ROUND(AVG(d.wteq_pct_median)::numeric, 1) as avg_basin_index,
    ROUND(MIN(d.wteq_pct_median)::numeric, 1) as min_basin_index,
    ROUND(MAX(d.wteq_pct_median)::numeric, 1) as max_basin_index,
    ROUND(AVG(d.snwd_value)::numeric, 1) as avg_snow_depth
FROM stations s
JOIN daily_observations d ON s.id = d.station_id
GROUP BY s.state_code, d.observation_date;

COMMENT ON VIEW vw_snotel_state_summary IS 'Daily state-level snowpack summary';

-- ============================================================================
-- FUNCTION: Get observations for any date (for SQL View in GeoServer)
-- ============================================================================
CREATE OR REPLACE FUNCTION get_snotel_by_date(target_date DATE)
RETURNS TABLE (
    station_id INTEGER,
    triplet VARCHAR(50),
    station_name VARCHAR(100),
    state_code VARCHAR(2),
    county_name VARCHAR(100),
    huc VARCHAR(12),
    elevation DOUBLE PRECISION,
    observation_date DATE,
    wteq_value DOUBLE PRECISION,
    wteq_median DOUBLE PRECISION,
    basin_index DOUBLE PRECISION,
    snow_depth DOUBLE PRECISION,
    precip DOUBLE PRECISION,
    temp_max DOUBLE PRECISION,
    temp_min DOUBLE PRECISION,
    condition_class TEXT,
    geom geometry
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.id as station_id,
        s.triplet,
        s.name as station_name,
        s.state_code,
        s.county_name,
        s.huc,
        s.elevation,
        d.observation_date,
        d.wteq_value,
        d.wteq_median,
        d.wteq_pct_median as basin_index,
        d.snwd_value as snow_depth,
        d.prec_value as precip,
        d.tmax_value as temp_max,
        d.tmin_value as temp_min,
        CASE 
            WHEN d.wteq_pct_median IS NULL THEN 'no-data'
            WHEN d.wteq_pct_median < 25 THEN 'critical'
            WHEN d.wteq_pct_median < 50 THEN 'low'
            WHEN d.wteq_pct_median < 75 THEN 'below-normal'
            WHEN d.wteq_pct_median < 90 THEN 'near-normal'
            WHEN d.wteq_pct_median < 110 THEN 'normal'
            WHEN d.wteq_pct_median < 150 THEN 'above-normal'
            ELSE 'high'
        END as condition_class,
        s.geom
    FROM stations s
    JOIN daily_observations d ON s.id = d.station_id
    WHERE d.observation_date = target_date
      AND s.geom IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_snotel_by_date IS 'Get SNOTEL observations for a specific date';

-- ============================================================================
-- GEOSERVER SQL VIEW CONFIGURATION
-- ============================================================================
-- In GeoServer, create a SQL View layer with this query:
--
-- SELECT * FROM get_snotel_by_date('%date%'::date)
--
-- Parameters:
--   Name: date
--   Default value: (SELECT MAX(observation_date) FROM daily_observations)
--   Validation regex: ^[0-9]{4}-[0-9]{2}-[0-9]{2}$
--
-- This allows WFS requests like:
--   .../wfs?service=WFS&request=GetFeature&typeName=snotel_by_date&viewparams=date:2025-01-15
-- ============================================================================

-- ============================================================================
-- PERMISSIONS (uncomment and adjust for your GeoServer user)
-- ============================================================================
-- CREATE USER geoserver WITH PASSWORD 'your_password';
-- GRANT CONNECT ON DATABASE water_risk TO geoserver;
-- GRANT USAGE ON SCHEMA public TO geoserver;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO geoserver;
-- GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO geoserver;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO geoserver;

-- ============================================================================
-- VERIFICATION
-- ============================================================================
SELECT 'Views created successfully!' as status;

SELECT 
    table_name as layer_name,
    'spatial' as layer_type
FROM information_schema.views 
WHERE table_schema = 'public' 
  AND table_name IN ('vw_snotel_stations', 'vw_snotel_latest')
UNION ALL
SELECT 
    matviewname as layer_name,
    'spatial (materialized)' as layer_type
FROM pg_matviews
WHERE schemaname = 'public'
  AND matviewname = 'mv_snotel_latest'
ORDER BY layer_name;

