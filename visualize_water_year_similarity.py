"""
Visualize Water Year Similarity Analysis

Creates multi-panel plots comparing a target water year against
the most similar years identified by each similarity method.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine

from config import DATABASE_URL
from water_year_similarity import (
    extract_station_water_years,
    compute_all_similarities,
    normalize_to_365_days,
    get_available_stations,
)


# Color palette - distinctive colors for each year
COLORS = {
    'target': '#E63946',  # Red for target year
    'similar': [
        '#457B9D',  # Steel blue
        '#2A9D8F',  # Teal
        '#E9C46A',  # Gold
        '#F4A261',  # Orange
    ]
}


def create_comparison_plot(
    wy_data: pd.DataFrame,
    target_wy: int,
    similar_years: list[int],
    method_name: str,
    scores: dict,
    ax: plt.Axes,
) -> None:
    """Create a single comparison plot on the given axes."""
    
    # Create x-axis as dates (using a reference non-leap year for display)
    ref_year = 2000
    dates = [datetime(ref_year - 1, 10, 1) + timedelta(days=i) for i in range(365)]
    
    # Helper to normalize series to 365 days
    def get_normalized_series(wy):
        if wy not in wy_data.columns:
            return np.full(365, np.nan)
        series = wy_data[wy].values
        if len(series) == 366:
            return normalize_to_365_days(series, is_leap_year=True)
        elif len(series) < 365:
            return np.pad(series, (0, 365 - len(series)), constant_values=np.nan)
        return series[:365]
    
    # Plot similar years first (so target is on top)
    for i, wy in enumerate(similar_years[:4]):
        series = get_normalized_series(wy)
        score = scores.get(wy, 0)
        
        # Format score for label
        if method_name in ['DTW', 'Euclidean']:
            score_str = f"{score:.2f}"
        else:
            score_str = f"{score:.3f}"
        
        ax.plot(
            dates, series,
            color=COLORS['similar'][i],
            linewidth=1.5,
            alpha=0.8,
            label=f'WY {wy} ({score_str})'
        )
    
    # Plot target year prominently
    target_series = get_normalized_series(target_wy)
    ax.plot(
        dates, target_series,
        color=COLORS['target'],
        linewidth=2.5,
        label=f'WY {target_wy} (target)',
        zorder=10
    )
    
    # Formatting
    ax.set_title(f'{method_name}', fontsize=12, fontweight='bold')
    ax.set_ylabel('SWE (inches)', fontsize=10)
    
    # Format x-axis as months
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(dates[0], dates[-1])
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)


def visualize_all_methods(
    station_triplet: str,
    target_wy: int = 2020,
    output_dir: str = "data/processed",
    min_wy: int = 1996,
    max_wy: int = 2025,
) -> str:
    """
    Create a multi-panel visualization comparing all similarity methods.
    
    Args:
        station_triplet: Station identifier (e.g., "473:CA:SNTL")
        target_wy: Water year to analyze
        output_dir: Directory to save the output
        min_wy, max_wy: Range of water years
    
    Returns:
        Path to the saved figure
    """
    print(f"Generating similarity visualization for {station_triplet}, WY {target_wy}")
    
    engine = create_engine(DATABASE_URL)
    
    # Get station name for title
    stations = get_available_stations(engine)
    station_info = stations[stations['triplet'] == station_triplet]
    if not station_info.empty:
        station_name = station_info.iloc[0]['name']
        state = station_info.iloc[0]['state_code']
    else:
        station_name = station_triplet
        state = ""
    
    # Extract water year data
    print("  Extracting water year data...")
    wy_data = extract_station_water_years(engine, station_triplet, min_wy, max_wy)
    
    if wy_data.empty:
        print("  ERROR: No data found")
        return None
    
    # Filter to water years with good coverage
    coverage = wy_data.notna().sum() / len(wy_data)
    valid_wys = coverage[coverage > 0.7].index.tolist()
    wy_data = wy_data[valid_wys]
    
    if target_wy not in wy_data.columns:
        print(f"  ERROR: Target WY {target_wy} not available")
        return None
    
    print(f"  Computing similarities across {len(wy_data.columns)} water years...")
    
    # Compute all similarities
    similarities = compute_all_similarities(
        wy_data, target_wy,
        methods=["dtw", "lcss", "correlation", "euclidean", "edm"]
    )
    
    # Define methods and their characteristics
    methods = [
        {
            'name': 'Dynamic Time Warping (DTW)',
            'column': 'dtw_distance',
            'ascending': True,  # Lower distance = more similar
            'description': 'Allows timing shifts'
        },
        {
            'name': 'Euclidean Distance',
            'column': 'euclidean_distance', 
            'ascending': True,
            'description': 'Point-by-point comparison'
        },
        {
            'name': 'Pearson Correlation',
            'column': 'correlation',
            'ascending': False,  # Higher correlation = more similar
            'description': 'Shape similarity'
        },
        {
            'name': 'LCSS (Longest Common Subsequence)',
            'column': 'lcss_similarity',
            'ascending': False,
            'description': 'Robust to gaps'
        },
        {
            'name': 'EDM (State-Space Similarity)',
            'column': 'edm_similarity',
            'ascending': False,
            'description': 'Trajectory similarity'
        },
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    # Generate plot for each method
    for i, method in enumerate(methods):
        col = method['column']
        if col not in similarities.columns:
            continue
        
        # Get top 4 similar years
        sorted_sim = similarities[col].sort_values(ascending=method['ascending'])
        top_years = sorted_sim.head(4).index.tolist()
        scores = sorted_sim.head(4).to_dict()
        
        create_comparison_plot(
            wy_data, target_wy, top_years,
            method['name'], scores, axes[i]
        )
        
        # Add description
        axes[i].text(
            0.02, 0.98, method['description'],
            transform=axes[i].transAxes,
            fontsize=9, style='italic',
            verticalalignment='top',
            color='gray'
        )
    
    # Hide the 6th subplot (we only have 5 methods)
    axes[5].axis('off')
    
    # Add summary text in the empty subplot
    summary_text = f"""
    Water Year Similarity Analysis
    
    Station: {station_name} ({state})
    Target: Water Year {target_wy}
    
    Each panel shows WY {target_wy} (red) compared
    to the 4 most similar years identified by
    each similarity method.
    
    Method Interpretations:
    • DTW: Similar patterns, timing may differ
    • Euclidean: Similar daily values
    • Correlation: Similar shape (not magnitude)
    • LCSS: Similar despite data gaps
    • EDM: Similar seasonal dynamics
    """
    axes[5].text(
        0.1, 0.9, summary_text,
        transform=axes[5].transAxes,
        fontsize=11,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6')
    )
    
    # Main title
    fig.suptitle(
        f'Water Year {target_wy} Similarity Analysis\n{station_name} ({station_triplet})',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"wy_similarity_{station_triplet.replace(':', '_')}_wy{target_wy}.png"
    filepath = output_path / filename
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved to: {filepath}")
    
    plt.close()
    
    return str(filepath)


def visualize_single_method(
    station_triplet: str,
    target_wy: int,
    method: str = "dtw",
    n_similar: int = 4,
    output_dir: str = "data/processed",
) -> str:
    """Create a single focused plot for one similarity method."""
    
    engine = create_engine(DATABASE_URL)
    
    # Get station info
    stations = get_available_stations(engine)
    station_info = stations[stations['triplet'] == station_triplet]
    station_name = station_info.iloc[0]['name'] if not station_info.empty else station_triplet
    
    # Extract data
    wy_data = extract_station_water_years(engine, station_triplet)
    
    # Filter valid years
    coverage = wy_data.notna().sum() / len(wy_data)
    valid_wys = coverage[coverage > 0.7].index.tolist()
    wy_data = wy_data[valid_wys]
    
    # Compute similarities
    similarities = compute_all_similarities(wy_data, target_wy, methods=[method])
    
    # Method config
    method_config = {
        'dtw': ('dtw_distance', True, 'Dynamic Time Warping'),
        'euclidean': ('euclidean_distance', True, 'Euclidean Distance'),
        'correlation': ('correlation', False, 'Pearson Correlation'),
        'lcss': ('lcss_similarity', False, 'LCSS'),
        'edm': ('edm_similarity', False, 'EDM State-Space'),
    }
    
    col, ascending, method_name = method_config[method]
    sorted_sim = similarities[col].sort_values(ascending=ascending)
    top_years = sorted_sim.head(n_similar).index.tolist()
    scores = sorted_sim.head(n_similar).to_dict()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    create_comparison_plot(wy_data, target_wy, top_years, method_name, scores, ax)
    
    ax.set_title(
        f'{method_name}: WY {target_wy} vs. Most Similar Years\n{station_name} ({station_triplet})',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('Date', fontsize=11)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"wy_similarity_{method}_{station_triplet.replace(':', '_')}_wy{target_wy}.png"
    filepath = output_path / filename
    
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved to: {filepath}")
    
    plt.close()
    
    return str(filepath)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Water Year Similarity")
    parser.add_argument("--station", type=str, default="473:CA:SNTL",
                       help="Station triplet")
    parser.add_argument("--target-wy", type=int, default=2020,
                       help="Target water year")
    parser.add_argument("--method", type=str, default=None,
                       help="Single method to visualize (dtw, euclidean, correlation, lcss, edm)")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                       help="Output directory")
    
    args = parser.parse_args()
    
    if args.method:
        visualize_single_method(
            args.station,
            args.target_wy,
            args.method,
            output_dir=args.output_dir,
        )
    else:
        visualize_all_methods(
            args.station,
            args.target_wy,
            output_dir=args.output_dir,
        )

