"""
EDM (Empirical Dynamic Modeling) Similarity Analysis

Implements proper EDM-based similarity following Sugihara et al. methods:
1. Optimal embedding dimension (E) selection via prediction skill
2. Optimal time delay (tau) selection via autocorrelation decay  
3. Cross-prediction similarity between water years
4. State-space trajectory comparison

References:
- Sugihara & May (1990) - Nonlinear forecasting as a way of distinguishing chaos
- Sugihara et al. (2012) - Detecting Causality in Complex Ecosystems (CCM)
- pyEDM documentation: https://github.com/SugiharaLab/pyEDM
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# OPTIMAL PARAMETER SELECTION
# =============================================================================

def find_optimal_embedding_dim(
    series: np.ndarray,
    max_E: int = 10,
    tau: int = 1,
    lib_fraction: float = 0.6,
    pred_fraction: float = 0.3,
) -> Tuple[int, float]:
    """Find optimal embedding dimension E using prediction skill.
    
    Uses pyEDM's EmbedDimension to test E=1 to max_E and returns
    the E that maximizes prediction correlation (rho).
    
    This follows the Sugihara method: the embedding dimension that
    best captures the attractor structure will have highest prediction skill.
    
    Args:
        series: Time series to analyze (should be normalized)
        max_E: Maximum embedding dimension to test
        tau: Time delay for embedding (default 1)
        lib_fraction: Fraction of data for library (training)
        pred_fraction: Fraction of data for prediction (testing)
    
    Returns:
        Tuple of (optimal_E, max_rho)
    """
    try:
        import pyEDM
    except ImportError:
        # Fallback: use mutual information heuristic
        return _find_E_heuristic(series, max_E)
    
    n = len(series)
    lib_end = int(n * lib_fraction)
    pred_start = lib_end + 1
    pred_end = min(pred_start + int(n * pred_fraction), n)
    
    if pred_end - pred_start < 10:
        return 3, np.nan  # Default if not enough data
    
    # Prepare DataFrame for pyEDM
    df = pd.DataFrame({
        'time': np.arange(1, n + 1),
        'value': series
    })
    
    try:
        result = pyEDM.EmbedDimension(
            dataFrame=df,
            lib=f'1 {lib_end}',
            pred=f'{pred_start} {pred_end}',
            columns='value',
            target='value',
            maxE=max_E,
            Tp=1,
            tau=-tau,  # Negative tau uses pyEDM convention
        )
        
        # Find E with maximum rho
        valid = result[result['rho'].notna()]
        if len(valid) == 0:
            return 3, np.nan
        
        optimal_idx = valid['rho'].idxmax()
        optimal_E = int(valid.loc[optimal_idx, 'E'])
        max_rho = valid.loc[optimal_idx, 'rho']
        
        return optimal_E, max_rho
    
    except Exception:
        return 3, np.nan


def _find_E_heuristic(series: np.ndarray, max_E: int = 10) -> Tuple[int, float]:
    """Heuristic E selection using false nearest neighbors concept.
    
    Fallback when pyEDM is not available.
    """
    # Simple heuristic: E ~ log2(n) but bounded
    n = len(series)
    E = min(max(2, int(np.log2(n) - 2)), max_E)
    return E, np.nan


def find_optimal_tau(
    series: np.ndarray,
    max_tau: int = 30,
    method: str = 'autocorr'
) -> int:
    """Find optimal time delay tau for embedding.
    
    Methods:
    - 'autocorr': First minimum of autocorrelation (decorrelation time)
    - 'mi': First minimum of mutual information (nonlinear decorrelation)
    
    Args:
        series: Time series to analyze
        max_tau: Maximum tau to consider
        method: 'autocorr' or 'mi'
    
    Returns:
        Optimal tau value
    """
    if method == 'autocorr':
        return _find_tau_autocorr(series, max_tau)
    elif method == 'mi':
        return _find_tau_mi(series, max_tau)
    else:
        return 1


def _find_tau_autocorr(series: np.ndarray, max_tau: int = 30) -> int:
    """Find tau as first zero crossing or 1/e decay of autocorrelation."""
    n = len(series)
    series_centered = series - np.mean(series)
    var = np.var(series)
    
    if var == 0:
        return 1
    
    # Compute autocorrelation for different lags
    autocorr = []
    for tau in range(max_tau + 1):
        if tau >= n:
            break
        corr = np.sum(series_centered[:n-tau] * series_centered[tau:]) / ((n - tau) * var)
        autocorr.append(corr)
    
    autocorr = np.array(autocorr)
    
    # Find first crossing below 1/e (~0.37) or zero
    threshold = 1 / np.e
    for i, ac in enumerate(autocorr[1:], 1):
        if ac < threshold:
            return i
        if ac < 0:
            return i
    
    return max_tau


def _find_tau_mi(series: np.ndarray, max_tau: int = 30, bins: int = 20) -> int:
    """Find tau as first local minimum of mutual information."""
    n = len(series)
    
    # Discretize series for MI calculation
    bin_edges = np.linspace(series.min(), series.max(), bins + 1)
    digitized = np.digitize(series, bin_edges[:-1])
    
    mi_values = []
    for tau in range(max_tau + 1):
        if tau >= n - 1:
            break
        
        x = digitized[:n-tau]
        y = digitized[tau:]
        
        # Compute joint and marginal distributions
        joint_hist = np.histogram2d(x, y, bins=bins)[0]
        joint_prob = joint_hist / np.sum(joint_hist)
        
        px = np.sum(joint_prob, axis=1)
        py = np.sum(joint_prob, axis=0)
        
        # Mutual information
        mi = 0
        for i in range(bins):
            for j in range(bins):
                if joint_prob[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (px[i] * py[j]))
        
        mi_values.append(mi)
    
    mi_values = np.array(mi_values)
    
    # Find first local minimum
    for i in range(1, len(mi_values) - 1):
        if mi_values[i] < mi_values[i-1] and mi_values[i] < mi_values[i+1]:
            return i
    
    return _find_tau_autocorr(series, max_tau)  # Fallback


# =============================================================================
# STATE-SPACE EMBEDDING
# =============================================================================

def create_embedding(
    series: np.ndarray,
    E: int,
    tau: int = 1,
) -> np.ndarray:
    """Create time-delay embedding of a series.
    
    Implements Takens' embedding theorem:
    x(t) → [x(t), x(t-τ), x(t-2τ), ..., x(t-(E-1)τ)]
    
    Args:
        series: 1D time series
        E: Embedding dimension
        tau: Time delay
    
    Returns:
        2D array of shape (n_points, E) representing the attractor
    """
    n = len(series)
    n_points = n - (E - 1) * tau
    
    if n_points <= 0:
        raise ValueError(f"Series too short for E={E}, tau={tau}")
    
    embedding = np.zeros((n_points, E))
    for i in range(E):
        start_idx = (E - 1 - i) * tau
        embedding[:, i] = series[start_idx:start_idx + n_points]
    
    return embedding


# =============================================================================
# EDM SIMILARITY METHODS
# =============================================================================

def edm_cross_prediction_similarity(
    series1: np.ndarray,
    series2: np.ndarray,
    E: Optional[int] = None,
    tau: Optional[int] = None,
) -> float:
    """Compute similarity using cross-prediction skill.
    
    Uses series1's attractor to predict series2 via Simplex projection.
    If series1 and series2 have similar dynamics, cross-prediction
    should be successful.
    
    This is inspired by CCM (Convergent Cross Mapping) but simplified
    for similarity rather than causality detection.
    
    Args:
        series1, series2: Time series to compare
        E: Embedding dimension (auto-detected if None)
        tau: Time delay (auto-detected if None)
    
    Returns:
        Similarity score in [0, 1]. Higher = more similar dynamics.
    """
    try:
        import pyEDM
    except ImportError:
        return _state_space_similarity_fallback(series1, series2, E or 3, tau or 7)
    
    # Normalize series
    s1 = (series1 - np.mean(series1)) / (np.std(series1) + 1e-10)
    s2 = (series2 - np.mean(series2)) / (np.std(series2) + 1e-10)
    
    # Find optimal parameters if not provided
    if E is None:
        E, _ = find_optimal_embedding_dim(s1)
    if tau is None:
        tau = find_optimal_tau(s1)
    
    n = len(s1)
    lib_end = int(n * 0.6)
    pred_start = lib_end + 1
    pred_end = n
    
    # Combine series: use s1 as library, predict s2
    # Create merged dataframe
    df = pd.DataFrame({
        'time': np.arange(1, n + 1),
        'lib_series': s1,
        'pred_series': s2,
    })
    
    try:
        # Use Simplex with s1 as library to predict s2's values
        result = pyEDM.Simplex(
            dataFrame=df,
            lib=f'1 {lib_end}',
            pred=f'{pred_start} {pred_end}',
            columns='lib_series',
            target='pred_series',  # Cross-predict s2 using s1's dynamics
            E=E,
            tau=-tau,
            Tp=1,
        )
        
        # Compute prediction skill
        valid = result[['Observations', 'Predictions']].dropna()
        if len(valid) < 5:
            return _state_space_similarity_fallback(s1, s2, E, tau)
        
        rho = np.corrcoef(valid['Observations'], valid['Predictions'])[0, 1]
        
        # Convert correlation to similarity [0, 1]
        similarity = (rho + 1) / 2
        return max(0, min(1, similarity))
    
    except Exception:
        return _state_space_similarity_fallback(s1, s2, E, tau)


def edm_attractor_similarity(
    series1: np.ndarray,
    series2: np.ndarray,
    E: Optional[int] = None,
    tau: Optional[int] = None,
) -> float:
    """Compute similarity by comparing attractor geometry.
    
    Creates state-space embeddings of both series and measures
    how well the trajectories overlap using nearest neighbor distances.
    
    If two water years have similar dynamics, their attractors
    should occupy similar regions of state space.
    
    Args:
        series1, series2: Time series to compare
        E: Embedding dimension (auto-detected if None)
        tau: Time delay (auto-detected if None)
    
    Returns:
        Similarity score in [0, 1]. Higher = more similar dynamics.
    """
    # Normalize series
    s1 = (series1 - np.mean(series1)) / (np.std(series1) + 1e-10)
    s2 = (series2 - np.mean(series2)) / (np.std(series2) + 1e-10)
    
    # Find optimal parameters if not provided
    if E is None:
        E, _ = find_optimal_embedding_dim(s1)
    if tau is None:
        tau = find_optimal_tau(s1)
    
    return _state_space_similarity_fallback(s1, s2, E, tau)


def _state_space_similarity_fallback(
    series1: np.ndarray,
    series2: np.ndarray,
    E: int = 3,
    tau: int = 7,
) -> float:
    """Compute attractor similarity without pyEDM.
    
    Creates time-delay embeddings and compares trajectories using
    bidirectional nearest neighbor distances.
    """
    try:
        emb1 = create_embedding(series1, E, tau)
        emb2 = create_embedding(series2, E, tau)
    except ValueError:
        return np.nan
    
    # Compute pairwise distances between all points
    distances = cdist(emb1, emb2, metric='euclidean')
    
    # Bidirectional nearest neighbor distance (Hausdorff-inspired)
    # For each point in emb1, find nearest in emb2
    nn_dist_1_to_2 = distances.min(axis=1)
    # For each point in emb2, find nearest in emb1
    nn_dist_2_to_1 = distances.min(axis=0)
    
    # Average of both directions
    mean_nn_dist = (nn_dist_1_to_2.mean() + nn_dist_2_to_1.mean()) / 2
    
    # Normalize by typical scale (std of embedded points)
    typical_scale = (np.std(emb1) + np.std(emb2)) / 2 * np.sqrt(E)
    
    # Convert distance to similarity
    similarity = 1 / (1 + mean_nn_dist / typical_scale)
    
    return similarity


def edm_prediction_skill_similarity(
    series1: np.ndarray,
    series2: np.ndarray,
    E: Optional[int] = None,
    Tp_max: int = 14,
) -> float:
    """Compare prediction skill decay profiles.
    
    Two series with similar dynamics should have similar
    prediction skill decay curves (rho vs Tp).
    
    Args:
        series1, series2: Time series to compare
        E: Embedding dimension (auto-detected if None)
        Tp_max: Maximum prediction horizon to test
    
    Returns:
        Similarity in [0, 1]. Higher = more similar forecast decay.
    """
    try:
        import pyEDM
    except ImportError:
        return np.nan
    
    s1 = (series1 - np.mean(series1)) / (np.std(series1) + 1e-10)
    s2 = (series2 - np.mean(series2)) / (np.std(series2) + 1e-10)
    
    if E is None:
        E, _ = find_optimal_embedding_dim(s1)
    
    n = len(s1)
    lib_end = int(n * 0.6)
    pred_start = lib_end + 1
    pred_end = n
    
    def get_prediction_decay(series):
        df = pd.DataFrame({
            'time': np.arange(1, n + 1),
            'value': series
        })
        
        try:
            result = pyEDM.PredictInterval(
                dataFrame=df,
                lib=f'1 {lib_end}',
                pred=f'{pred_start} {pred_end}',
                columns='value',
                target='value',
                E=E,
                maxTp=Tp_max,
            )
            return result['rho'].values
        except Exception:
            return np.full(Tp_max, np.nan)
    
    decay1 = get_prediction_decay(s1)
    decay2 = get_prediction_decay(s2)
    
    # Compare decay profiles
    valid_mask = ~(np.isnan(decay1) | np.isnan(decay2))
    if valid_mask.sum() < 3:
        return np.nan
    
    # Correlation of decay profiles
    rho = np.corrcoef(decay1[valid_mask], decay2[valid_mask])[0, 1]
    
    # Convert to similarity
    return (rho + 1) / 2


# =============================================================================
# MAIN SIMILARITY FUNCTION
# =============================================================================

def compute_edm_similarity(
    series1: np.ndarray,
    series2: np.ndarray,
    method: str = 'attractor',
    E: Optional[int] = None,
    tau: Optional[int] = None,
    optimize_params: bool = True,
) -> dict:
    """Compute EDM-based similarity with optimal parameter selection.
    
    Args:
        series1, series2: Time series to compare (raw values, will be normalized)
        method: Similarity method
            - 'attractor': State-space trajectory comparison
            - 'cross_prediction': Cross-prediction skill
            - 'prediction_decay': Compare forecast decay profiles
        E: Embedding dimension (auto-selected if None and optimize_params=True)
        tau: Time delay (auto-selected if None and optimize_params=True)
        optimize_params: Whether to auto-optimize E and tau
    
    Returns:
        Dictionary with:
            - 'similarity': Similarity score in [0, 1]
            - 'E': Embedding dimension used
            - 'tau': Time delay used
            - 'E_rho': Prediction skill at optimal E (if computed)
    """
    # Handle NaN values
    if np.isnan(series1).any() or np.isnan(series2).any():
        return {'similarity': np.nan, 'E': None, 'tau': None, 'E_rho': None}
    
    # Normalize series
    s1 = (series1 - np.mean(series1)) / (np.std(series1) + 1e-10)
    s2 = (series2 - np.mean(series2)) / (np.std(series2) + 1e-10)
    
    # Optimize parameters if requested
    E_rho = None
    if optimize_params:
        if E is None:
            E, E_rho = find_optimal_embedding_dim(s1)
        if tau is None:
            tau = find_optimal_tau(s1)
    else:
        E = E or 3
        tau = tau or 7
    
    # Compute similarity based on method
    if method == 'attractor':
        similarity = edm_attractor_similarity(s1, s2, E, tau)
    elif method == 'cross_prediction':
        similarity = edm_cross_prediction_similarity(s1, s2, E, tau)
    elif method == 'prediction_decay':
        similarity = edm_prediction_skill_similarity(s1, s2, E)
    else:
        similarity = edm_attractor_similarity(s1, s2, E, tau)
    
    return {
        'similarity': similarity,
        'E': E,
        'tau': tau,
        'E_rho': E_rho,
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from sqlalchemy import create_engine
    from config import DATABASE_URL
    from water_year_similarity import (
        extract_station_water_years,
        normalize_to_365_days,
        impute_missing_linear,
    )
    
    engine = create_engine(DATABASE_URL)
    
    # Get data
    wy_data = extract_station_water_years(engine, '473:CA:SNTL', 1996, 2025)
    coverage = wy_data.notna().sum() / len(wy_data)
    valid_wys = sorted(coverage[coverage > 0.7].index.tolist())
    wy_data = wy_data[valid_wys]
    
    def get_clean_series(wy):
        series = wy_data[wy].values.copy()
        if len(series) == 366:
            series = normalize_to_365_days(series, is_leap_year=True)
        elif len(series) < 365:
            series = np.pad(series, (0, 365 - len(series)), constant_values=np.nan)
        return impute_missing_linear(series[:365], max_gap=14)
    
    # Test with WY 2020 vs others
    target_wy = 2020
    s1 = get_clean_series(target_wy)
    
    print("=" * 70)
    print(f"EDM SIMILARITY ANALYSIS - Target: WY {target_wy}")
    print("=" * 70)
    
    # Find optimal parameters for target series
    print("\n1. OPTIMAL PARAMETER SELECTION")
    print("-" * 70)
    
    # Focus on snow season for better results
    snow_season = s1[50:200]
    E_opt, E_rho = find_optimal_embedding_dim(snow_season)
    tau_opt = find_optimal_tau(snow_season)
    
    print(f"   Optimal E: {E_opt} (prediction rho: {E_rho:.4f})")
    print(f"   Optimal tau: {tau_opt} days")
    
    # Compare to several years
    print("\n2. SIMILARITY TO OTHER YEARS")
    print("-" * 70)
    print(f"   {'WY':<8} {'Attractor':<12} {'Cross-Pred':<12} {'E':<6} {'tau':<6}")
    print("-" * 70)
    
    test_years = [2017, 2018, 2019, 2021, 2022]
    for wy in test_years:
        s2 = get_clean_series(wy)
        
        # Use snow season
        s1_snow = s1[50:200]
        s2_snow = s2[50:200]
        
        result_attr = compute_edm_similarity(s1_snow, s2_snow, method='attractor')
        result_xpred = compute_edm_similarity(s1_snow, s2_snow, method='cross_prediction')
        
        print(f"   {wy:<8} {result_attr['similarity']:.4f}       "
              f"{result_xpred['similarity']:.4f}       "
              f"{result_attr['E']:<6} {result_attr['tau']:<6}")
    
    print("\n✓ EDM similarity analysis complete")

