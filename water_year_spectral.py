"""
Spectral and Transform-Based Water Year Similarity Analysis

Methods that transform time series data into alternative representations:

1. **Fourier Transform (FFT)** - Frequency domain analysis
   - Captures periodic patterns (annual cycle, harmonics)
   - DFT coefficients as compact features
   - O(n log n) complexity

2. **Discrete Wavelet Transform (DWT)** - Multi-resolution analysis
   - Captures both frequency AND time information
   - Separates accumulation phase from melt phase
   - Excellent for non-stationary signals like SWE

3. **Symbolic Aggregate approXimation (SAX)** - Symbolic representation
   - Converts continuous series to discrete symbols
   - Enables fast string matching algorithms
   - Robust to noise

4. **Piecewise Aggregate Approximation (PAA)** - Segmented means
   - Simple but effective dimensionality reduction
   - Foundation for SAX
"""

import numpy as np
import pandas as pd
from scipy import fft, signal
from scipy.spatial.distance import euclidean, cosine
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# FOURIER TRANSFORM METHODS
# =============================================================================

def compute_fft_features(series: np.ndarray, n_coefficients: int = 20) -> dict:
    """
    Compute Fourier Transform features from a water year time series.
    
    The FFT decomposes the signal into frequency components:
    - Low frequencies: overall trend (accumulation, melt)
    - High frequencies: day-to-day variability, noise
    
    For SWE data, the first few coefficients capture most of the pattern.
    
    Args:
        series: 365-day SWE time series
        n_coefficients: Number of FFT coefficients to keep
    
    Returns:
        Dictionary with FFT features
    """
    # Handle missing values
    series_clean = series.copy()
    nan_mask = np.isnan(series_clean)
    if nan_mask.any():
        # Linear interpolation for NaN
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) < 10:
            return {'fft_coefficients': np.full(n_coefficients * 2, np.nan)}
        series_clean = np.interp(
            np.arange(len(series)),
            valid_idx,
            series[valid_idx]
        )
    
    # Remove mean (DC component) and normalize
    series_centered = series_clean - np.mean(series_clean)
    
    # Compute FFT
    fft_result = fft.fft(series_centered)
    
    # Take first n_coefficients (positive frequencies only)
    # Each coefficient has real and imaginary parts (or magnitude and phase)
    fft_truncated = fft_result[1:n_coefficients + 1]  # Skip DC
    
    # Extract magnitude and phase
    magnitudes = np.abs(fft_truncated)
    phases = np.angle(fft_truncated)
    
    features = {
        'fft_magnitudes': magnitudes,
        'fft_phases': phases,
        'fft_coefficients': np.concatenate([magnitudes, phases]),  # Combined
        'dominant_frequency': np.argmax(magnitudes) + 1,  # Which frequency dominates
        'total_power': np.sum(magnitudes ** 2),
        'spectral_entropy': _spectral_entropy(magnitudes),
    }
    
    return features


def _spectral_entropy(magnitudes: np.ndarray) -> float:
    """Compute spectral entropy (measure of frequency distribution)."""
    power = magnitudes ** 2
    power_norm = power / (power.sum() + 1e-10)
    entropy = -np.sum(power_norm * np.log(power_norm + 1e-10))
    return entropy


def fft_similarity(series1: np.ndarray, series2: np.ndarray, n_coefficients: int = 20) -> float:
    """
    Compute similarity between two series using FFT coefficients.
    
    Uses cosine similarity in the frequency domain.
    Higher values = more similar spectral content.
    """
    fft1 = compute_fft_features(series1, n_coefficients)
    fft2 = compute_fft_features(series2, n_coefficients)
    
    coef1 = fft1['fft_coefficients']
    coef2 = fft2['fft_coefficients']
    
    if np.isnan(coef1).any() or np.isnan(coef2).any():
        return np.nan
    
    # Cosine similarity
    return 1 - cosine(coef1, coef2)


def compute_power_spectrum(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using Welch's method.
    
    Returns frequencies and power at each frequency.
    """
    series_clean = np.nan_to_num(series, nan=0)
    frequencies, psd = signal.welch(series_clean, fs=1.0, nperseg=min(128, len(series)//4))
    return frequencies, psd


# =============================================================================
# WAVELET TRANSFORM METHODS
# =============================================================================

def compute_dwt_features(series: np.ndarray, wavelet: str = 'db4', level: int = 4) -> dict:
    """
    Compute Discrete Wavelet Transform features.
    
    DWT provides multi-resolution analysis:
    - Level 1: High-frequency details (daily noise)
    - Level 2-3: Medium-frequency (weekly patterns)
    - Level 4+: Low-frequency (seasonal trends)
    
    For SWE, the approximation coefficients at level 4 capture
    the overall accumulation/melt pattern.
    
    Args:
        series: SWE time series
        wavelet: Wavelet family ('db4' = Daubechies 4)
        level: Decomposition level
    
    Returns:
        Dictionary with wavelet features
    """
    try:
        import pywt
    except ImportError:
        return {'error': 'pywt not installed. Run: pip install PyWavelets'}
    
    # Handle NaN
    series_clean = series.copy()
    nan_mask = np.isnan(series_clean)
    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) < 10:
            return {'dwt_coefficients': np.array([])}
        series_clean = np.interp(
            np.arange(len(series)),
            valid_idx,
            series[valid_idx]
        )
    
    # Compute DWT
    coeffs = pywt.wavedec(series_clean, wavelet, level=level)
    
    # coeffs[0] = approximation (low freq trend)
    # coeffs[1:] = details at each level (high to low freq)
    
    features = {
        'approximation': coeffs[0],  # Overall shape
        'details': coeffs[1:],  # Multi-scale variations
        'dwt_coefficients': np.concatenate(coeffs),  # All coefficients
        'energy_by_level': [np.sum(c**2) for c in coeffs],
        'approximation_energy_ratio': np.sum(coeffs[0]**2) / sum(np.sum(c**2) for c in coeffs),
    }
    
    # Summary statistics per level
    features['level_means'] = [np.mean(np.abs(c)) for c in coeffs]
    features['level_stds'] = [np.std(c) for c in coeffs]
    
    return features


def dwt_similarity(series1: np.ndarray, series2: np.ndarray, wavelet: str = 'db4', level: int = 4) -> float:
    """
    Compute similarity using wavelet coefficients.
    
    Compares the approximation coefficients (overall shape)
    with optional weighting of detail coefficients.
    """
    try:
        import pywt
    except ImportError:
        return np.nan
    
    dwt1 = compute_dwt_features(series1, wavelet, level)
    dwt2 = compute_dwt_features(series2, wavelet, level)
    
    if 'error' in dwt1 or 'error' in dwt2:
        return np.nan
    
    # Compare approximation coefficients (most important)
    approx_sim = 1 - cosine(dwt1['approximation'], dwt2['approximation'])
    
    # Could also weight in detail similarities
    return approx_sim


def wavelet_decompose_and_visualize(series: np.ndarray, wavelet: str = 'db4', level: int = 4) -> dict:
    """
    Decompose series into wavelet components for visualization.
    
    Returns reconstructed signals at each level.
    """
    try:
        import pywt
    except ImportError:
        return {'error': 'pywt not installed'}
    
    # Handle NaN
    series_clean = np.nan_to_num(series, nan=0)
    
    # Decompose
    coeffs = pywt.wavedec(series_clean, wavelet, level=level)
    
    # Reconstruct at each level
    reconstructions = {}
    
    # Approximation only (trend)
    coeffs_approx = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    reconstructions['trend'] = pywt.waverec(coeffs_approx, wavelet)[:len(series)]
    
    # Each detail level
    for i in range(1, len(coeffs)):
        coeffs_detail = [np.zeros_like(coeffs[0])] + [np.zeros_like(c) for c in coeffs[1:]]
        coeffs_detail[i] = coeffs[i]
        reconstructions[f'detail_level_{i}'] = pywt.waverec(coeffs_detail, wavelet)[:len(series)]
    
    return reconstructions


# =============================================================================
# SYMBOLIC AGGREGATE APPROXIMATION (SAX)
# =============================================================================

def compute_paa(series: np.ndarray, n_segments: int = 12) -> np.ndarray:
    """
    Piecewise Aggregate Approximation (PAA).
    
    Divides series into n_segments and takes the mean of each.
    For water years, 12 segments = monthly averages.
    
    Args:
        series: Time series
        n_segments: Number of segments (12 = monthly)
    
    Returns:
        Array of segment means
    """
    series_clean = np.nan_to_num(series, nan=0)
    n = len(series_clean)
    segment_size = n // n_segments
    
    paa = np.zeros(n_segments)
    for i in range(n_segments):
        start = i * segment_size
        end = start + segment_size if i < n_segments - 1 else n
        paa[i] = np.mean(series_clean[start:end])
    
    return paa


def compute_sax(
    series: np.ndarray,
    n_segments: int = 12,
    alphabet_size: int = 5,
) -> str:
    """
    Symbolic Aggregate approXimation (SAX).
    
    Converts continuous time series to a string of symbols.
    Enables fast string matching and indexing.
    
    For water years:
    - 12 segments = monthly resolution
    - 5 symbols: a=very low, b=low, c=normal, d=high, e=very high
    
    Args:
        series: Time series
        n_segments: Number of PAA segments
        alphabet_size: Number of distinct symbols (3-10)
    
    Returns:
        SAX string representation
    """
    from scipy.stats import norm
    
    # Compute PAA
    paa = compute_paa(series, n_segments)
    
    # Z-normalize PAA
    if paa.std() > 0:
        paa_norm = (paa - paa.mean()) / paa.std()
    else:
        paa_norm = paa - paa.mean()
    
    # Define breakpoints for Gaussian distribution
    breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
    
    # Map to symbols
    alphabet = 'abcdefghij'[:alphabet_size]
    sax_string = ''
    
    for val in paa_norm:
        # Find which bin this value falls into
        idx = np.searchsorted(breakpoints, val)
        sax_string += alphabet[idx]
    
    return sax_string


def sax_distance(sax1: str, sax2: str) -> int:
    """
    Compute distance between two SAX strings.
    
    Simple: Hamming distance (number of different characters)
    """
    if len(sax1) != len(sax2):
        raise ValueError("SAX strings must be same length")
    
    return sum(c1 != c2 for c1, c2 in zip(sax1, sax2))


def sax_mindist(sax1: str, sax2: str, alphabet_size: int = 5, n: int = 365) -> float:
    """
    MINDIST: Lower-bounding distance for SAX.
    
    This distance lower-bounds the true Euclidean distance,
    making it useful for fast approximate nearest neighbor search.
    """
    from scipy.stats import norm
    
    if len(sax1) != len(sax2):
        raise ValueError("SAX strings must be same length")
    
    w = len(sax1)  # Number of segments
    
    # Breakpoints
    breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
    breakpoints = np.concatenate([[-np.inf], breakpoints, [np.inf]])
    
    # Distance lookup table
    def cell_distance(c1, c2):
        i1, i2 = ord(c1) - ord('a'), ord(c2) - ord('a')
        if abs(i1 - i2) <= 1:
            return 0
        else:
            return breakpoints[max(i1, i2)] - breakpoints[min(i1, i2) + 1]
    
    dist_sq = sum(cell_distance(c1, c2) ** 2 for c1, c2 in zip(sax1, sax2))
    
    return np.sqrt(n / w) * np.sqrt(dist_sq)


# =============================================================================
# COMBINED SPECTRAL FEATURE EXTRACTION
# =============================================================================

def extract_spectral_features(series: np.ndarray) -> dict:
    """
    Extract comprehensive spectral features from a water year.
    
    Combines FFT, wavelet, and SAX representations.
    """
    features = {}
    
    # FFT features
    fft_feats = compute_fft_features(series, n_coefficients=15)
    features['fft_magnitudes'] = fft_feats['fft_magnitudes']
    features['dominant_frequency'] = fft_feats['dominant_frequency']
    features['spectral_entropy'] = fft_feats['spectral_entropy']
    features['total_power'] = fft_feats['total_power']
    
    # Wavelet features (if available)
    try:
        dwt_feats = compute_dwt_features(series, level=4)
        if 'error' not in dwt_feats:
            features['dwt_energy_ratio'] = dwt_feats['approximation_energy_ratio']
            features['dwt_level_means'] = dwt_feats['level_means']
    except:
        pass
    
    # PAA (monthly averages)
    features['paa_monthly'] = compute_paa(series, n_segments=12)
    
    # SAX representation
    features['sax_string'] = compute_sax(series, n_segments=12, alphabet_size=5)
    
    return features


def compute_spectral_similarity_matrix(
    wy_data: pd.DataFrame,
    method: str = 'fft',
    n_coefficients: int = 20,
) -> pd.DataFrame:
    """
    Compute pairwise similarity matrix using spectral methods.
    
    Args:
        wy_data: Water year data (columns = years)
        method: 'fft', 'dwt', 'sax', or 'paa'
        n_coefficients: Number of coefficients for FFT
    
    Returns:
        Similarity matrix
    """
    water_years = list(wy_data.columns)
    n = len(water_years)
    
    if method == 'fft':
        # Extract FFT coefficients for all years
        features = {}
        for wy in water_years:
            fft_feats = compute_fft_features(wy_data[wy].values, n_coefficients)
            features[wy] = fft_feats['fft_coefficients']
        
        # Compute cosine similarity
        matrix = np.zeros((n, n))
        for i, wy1 in enumerate(water_years):
            for j, wy2 in enumerate(water_years):
                if i <= j:
                    if np.isnan(features[wy1]).any() or np.isnan(features[wy2]).any():
                        sim = np.nan
                    else:
                        sim = 1 - cosine(features[wy1], features[wy2])
                    matrix[i, j] = sim
                    matrix[j, i] = sim
    
    elif method == 'paa':
        # PAA comparison (very fast)
        features = {wy: compute_paa(wy_data[wy].values, 12) for wy in water_years}
        
        matrix = np.zeros((n, n))
        for i, wy1 in enumerate(water_years):
            for j, wy2 in enumerate(water_years):
                if i <= j:
                    dist = euclidean(features[wy1], features[wy2])
                    sim = 1 / (1 + dist)
                    matrix[i, j] = sim
                    matrix[j, i] = sim
    
    elif method == 'sax':
        # SAX distance
        features = {wy: compute_sax(wy_data[wy].values, 12, 5) for wy in water_years}
        
        matrix = np.zeros((n, n))
        for i, wy1 in enumerate(water_years):
            for j, wy2 in enumerate(water_years):
                if i <= j:
                    dist = sax_distance(features[wy1], features[wy2])
                    sim = 1 - dist / 12  # Normalize by max distance
                    matrix[i, j] = sim
                    matrix[j, i] = sim
    
    else:  # dwt
        try:
            import pywt
            features = {}
            for wy in water_years:
                dwt_feats = compute_dwt_features(wy_data[wy].values, level=4)
                features[wy] = dwt_feats.get('approximation', np.array([]))
            
            matrix = np.zeros((n, n))
            for i, wy1 in enumerate(water_years):
                for j, wy2 in enumerate(water_years):
                    if i <= j and len(features[wy1]) > 0 and len(features[wy2]) > 0:
                        sim = 1 - cosine(features[wy1], features[wy2])
                        matrix[i, j] = sim
                        matrix[j, i] = sim
        except ImportError:
            return pd.DataFrame()
    
    return pd.DataFrame(matrix, index=water_years, columns=water_years)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_spectral_analysis(series: np.ndarray, title: str = "Water Year") -> None:
    """Create visualization of spectral decomposition."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    days = np.arange(1, len(series) + 1)
    
    # Original series
    axes[0, 0].plot(days, series, 'b-', linewidth=1)
    axes[0, 0].set_title('Original Time Series')
    axes[0, 0].set_xlabel('Day of Water Year')
    axes[0, 0].set_ylabel('SWE (inches)')
    
    # FFT Power Spectrum
    fft_feats = compute_fft_features(series, 50)
    axes[0, 1].bar(range(1, len(fft_feats['fft_magnitudes']) + 1), 
                   fft_feats['fft_magnitudes'], color='steelblue')
    axes[0, 1].set_title('FFT Magnitude Spectrum')
    axes[0, 1].set_xlabel('Frequency Component')
    axes[0, 1].set_ylabel('Magnitude')
    
    # PAA representation
    paa = compute_paa(series, 12)
    months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 
              'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    axes[1, 0].bar(months, paa, color='teal')
    axes[1, 0].set_title('PAA (Monthly Averages)')
    axes[1, 0].set_ylabel('SWE (inches)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # SAX representation
    sax = compute_sax(series, 12, 5)
    sax_colors = {'a': '#d73027', 'b': '#fc8d59', 'c': '#fee090', 
                  'd': '#91bfdb', 'e': '#4575b4'}
    colors = [sax_colors[c] for c in sax]
    axes[1, 1].bar(months, [ord(c) - ord('a') + 1 for c in sax], color=colors)
    axes[1, 1].set_title(f'SAX Representation: "{sax}"')
    axes[1, 1].set_ylabel('Symbol (a=low, e=high)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Wavelet decomposition (if available)
    try:
        decomp = wavelet_decompose_and_visualize(series, level=4)
        axes[2, 0].plot(days, series, 'b-', alpha=0.3, label='Original')
        axes[2, 0].plot(days, decomp['trend'], 'r-', linewidth=2, label='Trend (Approx)')
        axes[2, 0].set_title('Wavelet Decomposition - Trend')
        axes[2, 0].legend()
        
        # Detail levels
        for i, (key, vals) in enumerate(decomp.items()):
            if 'detail' in key:
                axes[2, 1].plot(days, vals, alpha=0.7, label=key)
        axes[2, 1].set_title('Wavelet Details (Multi-scale)')
        axes[2, 1].legend(fontsize=8)
    except:
        axes[2, 0].text(0.5, 0.5, 'Wavelet: pip install PyWavelets', 
                        ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 1].text(0.5, 0.5, 'Wavelet: pip install PyWavelets',
                        ha='center', va='center', transform=axes[2, 1].transAxes)
    
    fig.suptitle(f'Spectral Analysis: {title}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    from sqlalchemy import create_engine
    from config import DATABASE_URL
    from water_year_similarity import extract_station_water_years
    import time
    
    engine = create_engine(DATABASE_URL)
    
    print("="*70)
    print("Spectral Methods for Water Year Similarity")
    print("="*70)
    
    # Get sample data
    wy_data = extract_station_water_years(engine, "473:CA:SNTL", 1996, 2025)
    coverage = wy_data.notna().sum() / len(wy_data)
    valid_wys = coverage[coverage > 0.7].index.tolist()
    wy_data = wy_data[valid_wys]
    
    print(f"\nDataset: {len(wy_data.columns)} water years")
    
    # Demo FFT
    print("\n" + "-"*70)
    print("1. FFT (Fourier Transform)")
    print("-"*70)
    
    start = time.time()
    fft_matrix = compute_spectral_similarity_matrix(wy_data, method='fft', n_coefficients=20)
    fft_time = time.time() - start
    
    print(f"Time: {fft_time:.3f}s")
    print(f"Most similar to WY 2020 (FFT):")
    print(fft_matrix.loc[2020].drop(2020).sort_values(ascending=False).head(5))
    
    # Demo PAA
    print("\n" + "-"*70)
    print("2. PAA (Piecewise Aggregate Approximation)")
    print("-"*70)
    
    start = time.time()
    paa_matrix = compute_spectral_similarity_matrix(wy_data, method='paa')
    paa_time = time.time() - start
    
    print(f"Time: {paa_time:.3f}s")
    print(f"Most similar to WY 2020 (PAA):")
    print(paa_matrix.loc[2020].drop(2020).sort_values(ascending=False).head(5))
    
    # Demo SAX
    print("\n" + "-"*70)
    print("3. SAX (Symbolic Aggregate approXimation)")
    print("-"*70)
    
    start = time.time()
    sax_matrix = compute_spectral_similarity_matrix(wy_data, method='sax')
    sax_time = time.time() - start
    
    print(f"Time: {sax_time:.3f}s")
    print(f"\nSAX representations (12 months, 5 symbols):")
    for wy in [2020, 2014, 2017]:
        sax_str = compute_sax(wy_data[wy].values, 12, 5)
        print(f"  WY {wy}: {sax_str}")
    
    print(f"\nMost similar to WY 2020 (SAX):")
    print(sax_matrix.loc[2020].drop(2020).sort_values(ascending=False).head(5))
    
    # Summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"FFT similarity matrix: {fft_time:.3f}s")
    print(f"PAA similarity matrix: {paa_time:.3f}s")
    print(f"SAX similarity matrix: {sax_time:.3f}s")
    
    # Save visualization
    print("\nGenerating spectral analysis plot...")
    fig = plot_spectral_analysis(wy_data[2020].values, "WY 2020")
    fig.savefig("data/processed/spectral_analysis_wy2020.png", dpi=150, bbox_inches='tight')
    print("Saved: data/processed/spectral_analysis_wy2020.png")

