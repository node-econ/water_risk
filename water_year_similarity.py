"""
Water Year Similarity Analysis

Compares the daily evolution of Snow Water Equivalent (SWE) across water years
using multiple similarity metrics:
    1. Dynamic Time Warping (DTW) - handles timing shifts
    2. Longest Common Subsequence (LCSS) - robust to missing values
    3. Pearson Correlation - shape similarity
    4. Empirical Dynamic Modeling (EDM) - state space similarity
    5. Delta methods (Δ SWE) - based on daily change, captures dynamics

Water Year: October 1 (Year N-1) through September 30 (Year N)
Example: WY 2025 = Oct 1, 2024 → Sep 30, 2025
"""

import warnings
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# DATA EXTRACTION
# =============================================================================

def get_water_year(d: date) -> int:
    """Return the water year for a given date.
    
    Water year N runs from Oct 1 of year N-1 to Sep 30 of year N.
    """
    if d.month >= 10:
        return d.year + 1
    return d.year


def get_day_of_water_year(d: date) -> int:
    """Return the day of water year (1-366) for a given date.
    
    Day 1 = October 1, Day 365/366 = September 30
    """
    wy = get_water_year(d)
    wy_start = date(wy - 1, 10, 1)
    return (d - wy_start).days + 1


def extract_station_water_years(
    engine,
    station_triplet: str,
    min_wy: int = 1996,
    max_wy: int = 2025,
) -> pd.DataFrame:
    """Extract SWE data organized by water year for a single station.
    
    Returns DataFrame with columns for each water year and rows for day-of-water-year.
    Index: 1-366 (day of water year)
    Columns: water year integers
    Values: SWE in inches
    """
    query = text("""
        SELECT 
            d.observation_date,
            d.wteq_value
        FROM daily_observations d
        JOIN stations s ON d.station_id = s.id
        WHERE s.triplet = :triplet
          AND d.observation_date >= :start_date
          AND d.observation_date <= :end_date
        ORDER BY d.observation_date
    """)
    
    start_date = date(min_wy - 1, 10, 1)
    end_date = date(max_wy, 9, 30)
    
    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={"triplet": station_triplet, "start_date": start_date, "end_date": end_date},
        )
    
    if df.empty:
        return pd.DataFrame()
    
    # Add water year and day of water year columns
    df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.date
    df["water_year"] = df["observation_date"].apply(get_water_year)
    df["day_of_wy"] = df["observation_date"].apply(get_day_of_water_year)
    
    # Pivot to wide format: rows = day_of_wy, columns = water_year
    pivot_df = df.pivot(index="day_of_wy", columns="water_year", values="wteq_value")
    
    # Ensure we have all days 1-366
    full_index = pd.Index(range(1, 367), name="day_of_wy")
    pivot_df = pivot_df.reindex(full_index)
    
    return pivot_df


def get_available_stations(engine) -> pd.DataFrame:
    """Get list of stations with their metadata."""
    query = text("""
        SELECT 
            s.triplet,
            s.name,
            s.state_code,
            s.elevation,
            COUNT(d.id) as obs_count,
            MIN(d.observation_date) as first_obs,
            MAX(d.observation_date) as last_obs
        FROM stations s
        JOIN daily_observations d ON s.id = d.station_id
        WHERE d.wteq_value IS NOT NULL
        GROUP BY s.triplet, s.name, s.state_code, s.elevation
        HAVING COUNT(d.id) > 365
        ORDER BY obs_count DESC
    """)
    
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


# =============================================================================
# PREPROCESSING
# =============================================================================

def normalize_to_365_days(series: np.ndarray, is_leap_year: bool = False) -> np.ndarray:
    """Interpolate a water year series to exactly 365 points.
    
    Handles leap years by interpolating day 366 data into 365 points.
    Missing values (NaN) are preserved during interpolation where possible.
    """
    if len(series) == 365 and not is_leap_year:
        return series
    
    # Original indices
    orig_days = np.arange(1, len(series) + 1)
    target_days = np.linspace(1, len(series), 365)
    
    # Find valid (non-NaN) values for interpolation
    valid_mask = ~np.isnan(series)
    
    if valid_mask.sum() < 10:  # Not enough data
        return np.full(365, np.nan)
    
    # Interpolate using valid values only
    interp_func = interp1d(
        orig_days[valid_mask],
        series[valid_mask],
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    
    return interp_func(target_days)


def impute_missing_linear(series: np.ndarray, max_gap: int = 14) -> np.ndarray:
    """Linearly interpolate missing values up to max_gap consecutive NaNs.
    
    Gaps larger than max_gap remain as NaN.
    """
    result = series.copy()
    n = len(result)
    
    # Find NaN segments
    is_nan = np.isnan(result)
    if not is_nan.any():
        return result
    
    # Get segments of consecutive NaNs
    nan_changes = np.diff(np.concatenate([[0], is_nan.astype(int), [0]]))
    nan_starts = np.where(nan_changes == 1)[0]
    nan_ends = np.where(nan_changes == -1)[0]
    
    for start, end in zip(nan_starts, nan_ends):
        gap_size = end - start
        if gap_size <= max_gap:
            # Get boundary values
            left_val = result[start - 1] if start > 0 else np.nan
            right_val = result[end] if end < n else np.nan
            
            if not np.isnan(left_val) and not np.isnan(right_val):
                # Linear interpolation
                result[start:end] = np.linspace(left_val, right_val, gap_size + 2)[1:-1]
            elif not np.isnan(left_val):
                result[start:end] = left_val  # Forward fill
            elif not np.isnan(right_val):
                result[start:end] = right_val  # Backward fill
    
    return result


def prepare_series_pair(
    series1: np.ndarray,
    series2: np.ndarray,
    impute: bool = True,
    normalize_length: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare two series for comparison.
    
    Args:
        series1, series2: Raw SWE time series (may have NaN, different lengths)
        impute: Whether to impute missing values
        normalize_length: Whether to normalize both to 365 days
    
    Returns:
        Tuple of processed series
    """
    s1, s2 = series1.copy(), series2.copy()
    
    if normalize_length:
        s1 = normalize_to_365_days(s1, len(series1) == 366)
        s2 = normalize_to_365_days(s2, len(series2) == 366)
    
    if impute:
        s1 = impute_missing_linear(s1)
        s2 = impute_missing_linear(s2)
    
    return s1, s2


# =============================================================================
# DELTA (DAILY CHANGE) TRANSFORMATION
# =============================================================================

def compute_delta(series: np.ndarray) -> np.ndarray:
    """Compute daily change (first derivative) of SWE series.
    
    Δ SWE captures the dynamics of accumulation and melt rather than
    absolute snowpack amounts. This is useful for finding years with
    similar weather patterns regardless of total snowpack.
    
    Args:
        series: SWE time series (365 or 366 days)
    
    Returns:
        Array of daily changes (length = len(series) - 1)
    """
    return np.diff(series)


def prepare_delta_series_pair(
    series1: np.ndarray,
    series2: np.ndarray,
    impute: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare two series as delta (daily change) for comparison.
    
    First normalizes and imputes the raw series, then computes delta.
    """
    # Prepare raw series first
    s1, s2 = prepare_series_pair(series1, series2, impute=impute, normalize_length=True)
    
    # Check for NaN
    if np.isnan(s1).any() or np.isnan(s2).any():
        # Try to impute remaining NaN
        s1 = impute_missing_linear(s1, max_gap=30)
        s2 = impute_missing_linear(s2, max_gap=30)
    
    # Compute delta
    d1 = compute_delta(s1)
    d2 = compute_delta(s2)
    
    return d1, d2


# =============================================================================
# DELTA-BASED SIMILARITY METHODS
# =============================================================================

def delta_correlation(series1: np.ndarray, series2: np.ndarray) -> float:
    """Correlation of daily changes (Δ SWE).
    
    Measures similarity in accumulation/melt dynamics rather than
    absolute snowpack amounts. A high correlation means both years
    had similar rates of snow gain/loss at similar times.
    
    Returns:
        Correlation coefficient in [-1, 1]. Higher = more similar dynamics.
    """
    d1, d2 = prepare_delta_series_pair(series1, series2)
    
    # Check for remaining NaN
    valid_mask = ~(np.isnan(d1) | np.isnan(d2))
    if valid_mask.sum() < 30:
        return np.nan
    
    corr, _ = stats.pearsonr(d1[valid_mask], d2[valid_mask])
    return corr


def delta_euclidean(series1: np.ndarray, series2: np.ndarray) -> float:
    """Euclidean distance of daily changes (Δ SWE).
    
    Measures point-by-point difference in accumulation/melt rates.
    
    Returns:
        Distance (lower = more similar dynamics).
    """
    d1, d2 = prepare_delta_series_pair(series1, series2)
    
    valid_mask = ~(np.isnan(d1) | np.isnan(d2))
    if valid_mask.sum() < 30:
        return np.nan
    
    return np.sqrt(np.mean((d1[valid_mask] - d2[valid_mask]) ** 2))


def delta_dtw(series1: np.ndarray, series2: np.ndarray) -> float:
    """DTW distance of daily changes (Δ SWE).
    
    Dynamic Time Warping on the change series. Finds years with
    similar dynamics even if timing is slightly shifted.
    
    Returns:
        Distance (lower = more similar dynamics).
    """
    try:
        from dtaidistance import dtw
        
        d1, d2 = prepare_delta_series_pair(series1, series2)
        
        if np.isnan(d1).any() or np.isnan(d2).any():
            return np.nan
        
        return dtw.distance(d1, d2)
    
    except ImportError:
        from tslearn.metrics import dtw as tslearn_dtw
        d1, d2 = prepare_delta_series_pair(series1, series2)
        if np.isnan(d1).any() or np.isnan(d2).any():
            return np.nan
        return tslearn_dtw(d1.reshape(-1, 1), d2.reshape(-1, 1))


def delta_fft_similarity(series1: np.ndarray, series2: np.ndarray, n_coefficients: int = 20) -> float:
    """FFT similarity of daily changes (Δ SWE).
    
    Compares the frequency content of the change series.
    Captures similar patterns of storm frequency and melt events.
    
    Returns:
        Similarity in [0, 1]. Higher = more similar spectral signature.
    """
    from scipy.spatial.distance import cosine
    
    d1, d2 = prepare_delta_series_pair(series1, series2)
    
    if np.isnan(d1).any() or np.isnan(d2).any():
        return np.nan
    
    # Compute FFT
    fft1 = np.fft.fft(d1 - np.mean(d1))[:n_coefficients]
    fft2 = np.fft.fft(d2 - np.mean(d2))[:n_coefficients]
    
    # Use magnitude (ignore phase for simplicity)
    mag1 = np.abs(fft1)
    mag2 = np.abs(fft2)
    
    # Cosine similarity
    return 1 - cosine(mag1, mag2)


# =============================================================================
# SIMILARITY METHODS
# =============================================================================

def dtw_distance(series1: np.ndarray, series2: np.ndarray) -> float:
    """Calculate Dynamic Time Warping distance between two series.
    
    DTW finds the optimal alignment between series, allowing for temporal
    stretching and compression. Good for comparing years where similar
    patterns may be shifted in time.
    
    Lower values = more similar
    """
    try:
        from dtaidistance import dtw
        
        # DTW requires no NaN values
        s1, s2 = prepare_series_pair(series1, series2, impute=True, normalize_length=True)
        
        # Check for remaining NaNs
        if np.isnan(s1).any() or np.isnan(s2).any():
            return np.nan
        
        return dtw.distance(s1, s2)
    
    except ImportError:
        # Fallback to tslearn
        from tslearn.metrics import dtw as tslearn_dtw
        s1, s2 = prepare_series_pair(series1, series2, impute=True, normalize_length=True)
        if np.isnan(s1).any() or np.isnan(s2).any():
            return np.nan
        return tslearn_dtw(s1.reshape(-1, 1), s2.reshape(-1, 1))


def lcss_similarity(
    series1: np.ndarray,
    series2: np.ndarray,
    epsilon: Optional[float] = None,
) -> float:
    """Calculate Longest Common Subsequence similarity.
    
    LCSS is robust to missing values and noise. It finds the longest
    subsequence where corresponding points are within epsilon of each other.
    
    Args:
        epsilon: Matching threshold. If None, uses 10% of series std dev.
    
    Returns:
        Similarity score in [0, 1]. Higher = more similar.
    """
    from tslearn.metrics import lcss
    
    # LCSS can handle some missing values but we still normalize length
    s1, s2 = prepare_series_pair(series1, series2, impute=False, normalize_length=True)
    
    # Replace NaN with very large values (will not match)
    s1_clean = np.nan_to_num(s1, nan=1e10)
    s2_clean = np.nan_to_num(s2, nan=1e10)
    
    # Determine epsilon based on data scale
    if epsilon is None:
        combined_std = np.nanstd(np.concatenate([series1, series2]))
        epsilon = max(0.5, combined_std * 0.1)  # 10% of std, min 0.5 inches
    
    # LCSS returns similarity in [0, 1]
    return lcss(s1_clean.reshape(-1, 1), s2_clean.reshape(-1, 1), eps=epsilon)


def correlation_similarity(series1: np.ndarray, series2: np.ndarray) -> float:
    """Calculate Pearson correlation between two series.
    
    Measures linear relationship / shape similarity. A correlation of 1.0
    means identical shapes (but possibly different magnitudes).
    
    Returns:
        Correlation coefficient in [-1, 1]. Higher = more similar shape.
    """
    s1, s2 = prepare_series_pair(series1, series2, impute=True, normalize_length=True)
    
    # Need valid pairs for correlation
    valid_mask = ~(np.isnan(s1) | np.isnan(s2))
    
    if valid_mask.sum() < 30:  # Need minimum data
        return np.nan
    
    corr, _ = stats.pearsonr(s1[valid_mask], s2[valid_mask])
    return corr


def euclidean_distance(series1: np.ndarray, series2: np.ndarray) -> float:
    """Calculate normalized Euclidean distance.
    
    Simple point-by-point distance. Sensitive to both shape and magnitude.
    
    Returns:
        Distance (lower = more similar). Normalized by series length.
    """
    s1, s2 = prepare_series_pair(series1, series2, impute=True, normalize_length=True)
    
    valid_mask = ~(np.isnan(s1) | np.isnan(s2))
    if valid_mask.sum() < 30:
        return np.nan
    
    # Root mean square difference
    return np.sqrt(np.mean((s1[valid_mask] - s2[valid_mask]) ** 2))


# =============================================================================
# EDM-BASED SIMILARITY (Sugihara et al.)
# =============================================================================

def edm_simplex_similarity(
    series1: np.ndarray,
    series2: np.ndarray,
    embedding_dim: Optional[int] = None,
    optimize_params: bool = True,
    method: str = 'attractor',
) -> float:
    """Calculate similarity using Empirical Dynamic Modeling concepts.
    
    Uses state-space reconstruction following Sugihara & May (1990) and
    the pyEDM library when available.
    
    The method:
    1. Auto-select optimal embedding dimension E using prediction skill
    2. Auto-select optimal time delay tau using autocorrelation decay
    3. Create time-delay embeddings (Takens' theorem)
    4. Compare attractor trajectories or cross-prediction skill
    
    Args:
        embedding_dim: Dimension for time-delay embedding (auto-optimized if None)
        optimize_params: Whether to auto-optimize E and tau (default True)
        method: 'attractor' (trajectory comparison) or 'cross_prediction'
    
    Returns:
        Similarity score in [0, 1]. Higher = more similar dynamics.
    """
    # Prepare series
    s1, s2 = prepare_series_pair(series1, series2, impute=True, normalize_length=True)
    
    if np.isnan(s1).any() or np.isnan(s2).any():
        return np.nan
    
    # Try to use the full EDM implementation
    try:
        from edm_similarity import compute_edm_similarity
        
        result = compute_edm_similarity(
            s1, s2,
            method=method,
            E=embedding_dim,
            optimize_params=optimize_params,
        )
        return result['similarity']
    
    except ImportError:
        # Fallback to simple implementation
        return _simplex_fallback(s1, s2, embedding_dim or 3)


def _simplex_fallback(
    series1: np.ndarray,
    series2: np.ndarray,
    E: int = 3,
    tau: int = 7,
) -> float:
    """Fallback simplex-style similarity without pyEDM.
    
    Creates time-delay embeddings and compares trajectories using
    nearest neighbor distances in the embedded space.
    
    Args:
        E: Embedding dimension
        tau: Time delay in days (default 7 = weekly)
    """
    from scipy.spatial.distance import cdist
    
    # Normalize
    s1 = (series1 - np.mean(series1)) / (np.std(series1) + 1e-10)
    s2 = (series2 - np.mean(series2)) / (np.std(series2) + 1e-10)
    
    # Create time-delay embeddings
    def embed(series, dim, delay):
        n = len(series) - (dim - 1) * delay
        if n <= 0:
            return None
        embedded = np.zeros((n, dim))
        for i in range(dim):
            start_idx = (dim - 1 - i) * delay
            embedded[:, i] = series[start_idx:start_idx + n]
        return embedded
    
    emb1 = embed(s1, E, tau)
    emb2 = embed(s2, E, tau)
    
    if emb1 is None or emb2 is None:
        return np.nan
    
    # Bidirectional nearest neighbor distances
    distances = cdist(emb1, emb2, metric="euclidean")
    nn_dist_1_to_2 = distances.min(axis=1)
    nn_dist_2_to_1 = distances.min(axis=0)
    mean_nn_dist = (nn_dist_1_to_2.mean() + nn_dist_2_to_1.mean()) / 2
    
    # Normalize by typical scale
    typical_scale = (np.std(emb1) + np.std(emb2)) / 2 * np.sqrt(E)
    similarity = 1 / (1 + mean_nn_dist / typical_scale)
    
    return similarity


def edm_with_optimal_params(
    series1: np.ndarray,
    series2: np.ndarray,
) -> dict:
    """EDM similarity with full parameter optimization and diagnostics.
    
    Returns detailed results including optimal E, tau, and prediction skill.
    
    Returns:
        Dictionary with 'similarity', 'E', 'tau', 'E_rho'
    """
    s1, s2 = prepare_series_pair(series1, series2, impute=True, normalize_length=True)
    
    if np.isnan(s1).any() or np.isnan(s2).any():
        return {'similarity': np.nan, 'E': None, 'tau': None, 'E_rho': None}
    
    try:
        from edm_similarity import compute_edm_similarity
        return compute_edm_similarity(s1, s2, method='attractor', optimize_params=True)
    except ImportError:
        return {
            'similarity': _simplex_fallback(s1, s2),
            'E': 3,
            'tau': 7,
            'E_rho': None
        }


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def compute_all_similarities(
    wy_data: pd.DataFrame,
    target_wy: int,
    methods: list[str] = ["dtw", "lcss", "correlation", "euclidean", "edm"],
) -> pd.DataFrame:
    """Compute similarity between target water year and all others.
    
    Args:
        wy_data: DataFrame with water years as columns, day_of_wy as index
        target_wy: Water year to compare against
        methods: List of methods to use. Available methods:
            - Raw SWE methods: "dtw", "lcss", "correlation", "euclidean", "edm"
            - Delta (Δ SWE) methods: "delta_correlation", "delta_euclidean", 
              "delta_dtw", "delta_fft"
    
    Returns:
        DataFrame with similarity scores for each water year and method
    """
    if target_wy not in wy_data.columns:
        raise ValueError(f"Target water year {target_wy} not in data")
    
    target_series = wy_data[target_wy].values
    other_wys = [wy for wy in wy_data.columns if wy != target_wy]
    
    results = []
    
    for wy in other_wys:
        comparison_series = wy_data[wy].values
        
        row = {"water_year": wy}
        
        # Raw SWE methods
        if "dtw" in methods:
            row["dtw_distance"] = dtw_distance(target_series, comparison_series)
        
        if "lcss" in methods:
            row["lcss_similarity"] = lcss_similarity(target_series, comparison_series)
        
        if "correlation" in methods:
            row["correlation"] = correlation_similarity(target_series, comparison_series)
        
        if "euclidean" in methods:
            row["euclidean_distance"] = euclidean_distance(target_series, comparison_series)
        
        if "edm" in methods:
            row["edm_similarity"] = edm_simplex_similarity(target_series, comparison_series)
        
        # Delta (Δ SWE) methods - based on daily change
        if "delta_correlation" in methods:
            row["delta_correlation"] = delta_correlation(target_series, comparison_series)
        
        if "delta_euclidean" in methods:
            row["delta_euclidean_distance"] = delta_euclidean(target_series, comparison_series)
        
        if "delta_dtw" in methods:
            row["delta_dtw_distance"] = delta_dtw(target_series, comparison_series)
        
        if "delta_fft" in methods:
            row["delta_fft_similarity"] = delta_fft_similarity(target_series, comparison_series)
        
        results.append(row)
    
    df = pd.DataFrame(results)
    df = df.set_index("water_year").sort_index()
    
    return df


def find_most_similar_years(
    wy_data: pd.DataFrame,
    target_wy: int,
    n_similar: int = 5,
    method: str = "dtw",
) -> pd.DataFrame:
    """Find the N most similar water years to the target.
    
    Args:
        wy_data: DataFrame with water years as columns
        target_wy: Water year to compare against
        n_similar: Number of similar years to return
        method: Similarity method to use. Options:
            - Raw: "dtw", "lcss", "correlation", "euclidean", "edm"
            - Delta: "delta_correlation", "delta_euclidean", "delta_dtw", "delta_fft"
    
    Returns:
        DataFrame with top N similar years and their scores
    """
    similarities = compute_all_similarities(wy_data, target_wy, methods=[method])
    
    # Determine if higher or lower is better
    distance_methods = ["dtw", "euclidean", "delta_euclidean", "delta_dtw"]
    
    if method in distance_methods:
        # Distance metrics - lower is better
        col_name = f"{method}_distance"
        sorted_df = similarities.sort_values(col_name, ascending=True)
    else:
        # Similarity metrics - higher is better
        if method in ["lcss", "edm", "delta_fft"]:
            col_name = f"{method}_similarity"
        elif method in ["correlation", "delta_correlation"]:
            col_name = method
        else:
            col_name = method
        sorted_df = similarities.sort_values(col_name, ascending=False)
    
    return sorted_df.head(n_similar)


def compute_pairwise_similarity_matrix(
    wy_data: pd.DataFrame,
    method: str = "correlation",
) -> pd.DataFrame:
    """Compute pairwise similarity matrix for all water years.
    
    Returns:
        Square DataFrame with similarity scores between all year pairs
    """
    water_years = list(wy_data.columns)
    n = len(water_years)
    
    matrix = np.zeros((n, n))
    
    method_func = {
        "dtw": dtw_distance,
        "lcss": lcss_similarity,
        "correlation": correlation_similarity,
        "euclidean": euclidean_distance,
        "edm": edm_simplex_similarity,
    }[method]
    
    for i, wy1 in enumerate(water_years):
        for j, wy2 in enumerate(water_years):
            if i == j:
                matrix[i, j] = 1.0 if method in ["lcss", "correlation", "edm"] else 0.0
            elif i < j:
                score = method_func(wy_data[wy1].values, wy_data[wy2].values)
                matrix[i, j] = score
                matrix[j, i] = score
    
    return pd.DataFrame(matrix, index=water_years, columns=water_years)


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_station_water_years(
    engine,
    station_triplet: str,
    target_wy: Optional[int] = None,
    min_wy: int = 1996,
    max_wy: int = 2025,
) -> dict:
    """Run complete water year similarity analysis for a station.
    
    Args:
        engine: SQLAlchemy database engine
        station_triplet: Station identifier (e.g., "302:OR:SNTL")
        target_wy: Specific water year to analyze (if None, uses most recent)
        min_wy, max_wy: Range of water years to include
    
    Returns:
        Dictionary with analysis results
    """
    print(f"Extracting data for {station_triplet}...")
    wy_data = extract_station_water_years(engine, station_triplet, min_wy, max_wy)
    
    if wy_data.empty:
        return {"error": "No data found for station"}
    
    # Filter to water years with sufficient data (>80% coverage)
    coverage = wy_data.notna().sum() / len(wy_data)
    valid_wys = coverage[coverage > 0.8].index.tolist()
    wy_data = wy_data[valid_wys]
    
    if len(wy_data.columns) < 2:
        return {"error": "Insufficient water years with data"}
    
    if target_wy is None:
        target_wy = max(wy_data.columns)
    elif target_wy not in wy_data.columns:
        return {"error": f"Target water year {target_wy} not available"}
    
    print(f"Analyzing {len(wy_data.columns)} water years...")
    print(f"Target: WY {target_wy}")
    print()
    
    # Compute similarities using all methods
    print("Computing similarities...")
    similarities = compute_all_similarities(
        wy_data, target_wy,
        methods=["dtw", "lcss", "correlation", "euclidean", "edm"]
    )
    
    # Find top similar years by each method
    results = {
        "station": station_triplet,
        "target_water_year": target_wy,
        "available_water_years": list(wy_data.columns),
        "data_coverage": coverage[valid_wys].to_dict(),
        "all_similarities": similarities,
        "top_similar": {},
    }
    
    print("\n" + "=" * 60)
    print(f"Most Similar Water Years to WY {target_wy}")
    print("=" * 60)
    
    for method, ascending in [
        ("dtw_distance", True),
        ("euclidean_distance", True),
        ("correlation", False),
        ("lcss_similarity", False),
        ("edm_similarity", False),
    ]:
        if method in similarities.columns:
            top_5 = similarities[method].sort_values(ascending=ascending).head(5)
            results["top_similar"][method] = top_5.to_dict()
            
            method_name = method.replace("_", " ").title()
            print(f"\n{method_name}:")
            for wy, score in top_5.items():
                print(f"  WY {wy}: {score:.4f}")
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_water_year_comparison(
    wy_data: pd.DataFrame,
    target_wy: int,
    similar_wys: list[int],
    output_path: Optional[str] = None,
) -> None:
    """Plot target water year against similar years.
    
    Creates a matplotlib figure showing SWE evolution comparison.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create x-axis as dates (using a reference year for display)
    ref_year = 2000  # Non-leap year for display
    dates = [datetime(ref_year - 1, 10, 1) + timedelta(days=i) for i in range(365)]
    
    # Plot similar years in gray
    for wy in similar_wys:
        if wy in wy_data.columns and wy != target_wy:
            series = wy_data[wy].values
            # Normalize to 365 days for plotting
            if len(series) > 365:
                series = normalize_to_365_days(series, is_leap_year=True)
            elif len(series) < 365:
                series = np.pad(series, (0, 365 - len(series)), constant_values=np.nan)
            ax.plot(dates, series[:365], color='gray', alpha=0.5, linewidth=1, label=f'WY {wy}')
    
    # Plot target year prominently
    target_series = wy_data[target_wy].values
    if len(target_series) > 365:
        target_series = normalize_to_365_days(target_series, is_leap_year=True)
    elif len(target_series) < 365:
        target_series = np.pad(target_series, (0, 365 - len(target_series)), constant_values=np.nan)
    ax.plot(dates, target_series[:365], color='#E63946', linewidth=2.5, label=f'WY {target_wy} (target)')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Snow Water Equivalent (inches)', fontsize=12)
    ax.set_title(f'Water Year {target_wy} vs. Similar Years', fontsize=14, fontweight='bold')
    
    # Format x-axis as months
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(dates[0], dates[-1])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_similarity_heatmap(
    similarity_matrix: pd.DataFrame,
    output_path: Optional[str] = None,
) -> None:
    """Plot heatmap of pairwise similarities between water years."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        similarity_matrix,
        annot=False,
        cmap='RdYlGn',
        center=0.5,
        ax=ax,
        square=True,
    )
    
    ax.set_title('Water Year Similarity Matrix (Correlation)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Water Year')
    ax.set_ylabel('Water Year')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    from config import DATABASE_URL
    
    parser = argparse.ArgumentParser(description="Water Year Similarity Analysis")
    parser.add_argument("--station", type=str, help="Station triplet (e.g., 302:OR:SNTL)")
    parser.add_argument("--target-wy", type=int, help="Target water year to compare")
    parser.add_argument("--list-stations", action="store_true", help="List available stations")
    parser.add_argument("--min-wy", type=int, default=1996, help="Minimum water year")
    parser.add_argument("--max-wy", type=int, default=2025, help="Maximum water year")
    
    args = parser.parse_args()
    
    engine = create_engine(DATABASE_URL)
    
    if args.list_stations:
        stations = get_available_stations(engine)
        print(f"\nAvailable stations ({len(stations)}):\n")
        print(stations.to_string())
    
    elif args.station:
        results = analyze_station_water_years(
            engine,
            args.station,
            target_wy=args.target_wy,
            min_wy=args.min_wy,
            max_wy=args.max_wy,
        )
        
        if "error" in results:
            print(f"Error: {results['error']}")
    
    else:
        # Demo with a sample station
        print("Running demo analysis...")
        stations = get_available_stations(engine)
        if not stations.empty:
            sample_station = stations.iloc[0]["triplet"]
            print(f"Using station: {sample_station}")
            results = analyze_station_water_years(engine, sample_station)

