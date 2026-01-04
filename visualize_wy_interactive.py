"""
Interactive Water Year Similarity Visualization

Creates interactive Plotly visualizations with:
1. Hover information showing exact SWE values and dates
2. Consensus ranking combining all similarity methods
3. HTML output for browser viewing
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine

from config import DATABASE_URL
from water_year_similarity import (
    extract_station_water_years,
    compute_all_similarities,
    normalize_to_365_days,
    get_available_stations,
    impute_missing_linear,
)


# =============================================================================
# CONSENSUS RANKING
# =============================================================================

def compute_consensus_ranking(
    similarities: pd.DataFrame,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Compute a consensus ranking by normalizing and combining all methods.
    
    Args:
        similarities: DataFrame with similarity scores from all methods
        weights: Optional dict of method weights (default: equal weights)
    
    Returns:
        DataFrame with consensus scores and ranks
    """
    if weights is None:
        weights = {
            'dtw_distance': 1.0,
            'euclidean_distance': 1.0,
            'correlation': 1.0,
            'lcss_similarity': 1.0,
            'edm_similarity': 1.0,
        }
    
    # Normalize each method to 0-1 scale (higher = more similar)
    normalized = pd.DataFrame(index=similarities.index)
    
    for col in similarities.columns:
        values = similarities[col].dropna()
        if len(values) == 0:
            continue
        
        min_val, max_val = values.min(), values.max()
        
        if max_val == min_val:
            normalized[col] = 0.5
        elif col in ['dtw_distance', 'euclidean_distance']:
            # Distance metrics: lower is better, so invert
            normalized[col] = 1 - (similarities[col] - min_val) / (max_val - min_val)
        else:
            # Similarity metrics: higher is better
            normalized[col] = (similarities[col] - min_val) / (max_val - min_val)
    
    # Compute weighted average
    total_weight = 0
    consensus = pd.Series(0.0, index=similarities.index)
    
    for col, weight in weights.items():
        if col in normalized.columns:
            # Handle NaN by treating as 0 contribution
            valid_mask = normalized[col].notna()
            consensus[valid_mask] += normalized[col][valid_mask] * weight
            total_weight += weight
    
    consensus = consensus / total_weight
    
    # Create result DataFrame
    result = pd.DataFrame({
        'consensus_score': consensus,
        'consensus_rank': consensus.rank(ascending=False),
    })
    
    # Add individual normalized scores
    for col in normalized.columns:
        result[f'{col}_normalized'] = normalized[col]
    
    return result.sort_values('consensus_score', ascending=False)


# =============================================================================
# INTERACTIVE PLOTLY VISUALIZATION
# =============================================================================

def create_interactive_comparison(
    wy_data: pd.DataFrame,
    target_wy: int,
    consensus_ranking: pd.DataFrame,
    station_name: str,
    station_triplet: str,
    n_similar: int = 5,
) -> go.Figure:
    """
    Create an interactive Plotly figure comparing water years.
    
    Features:
    - Hover to see exact SWE values and dates
    - Toggle years on/off in legend
    - Zoom and pan
    """
    
    # Create date axis (reference year for display)
    ref_year = 2000
    dates = [datetime(ref_year - 1, 10, 1) + timedelta(days=i) for i in range(365)]
    
    # Month labels for better display
    def day_to_date_str(day_idx):
        d = datetime(ref_year - 1, 10, 1) + timedelta(days=day_idx)
        return d.strftime('%b %d')
    
    # Helper to get normalized series
    def get_series(wy):
        if wy not in wy_data.columns:
            return np.full(365, np.nan)
        series = wy_data[wy].values.copy()
        if len(series) == 366:
            series = normalize_to_365_days(series, is_leap_year=True)
        elif len(series) < 365:
            series = np.pad(series, (0, 365 - len(series)), constant_values=np.nan)
        # Light imputation for smoother display
        series = impute_missing_linear(series[:365], max_gap=7)
        return series[:365]
    
    # Get top similar years from consensus
    top_years = consensus_ranking.head(n_similar).index.tolist()
    
    # Color scheme
    colors = {
        'target': '#E63946',  # Red
        'similar': [
            '#457B9D',  # Steel blue
            '#2A9D8F',  # Teal  
            '#E9C46A',  # Gold
            '#F4A261',  # Orange
            '#9B5DE5',  # Purple
        ]
    }
    
    fig = go.Figure()
    
    # Add similar years first
    for i, wy in enumerate(top_years):
        series = get_series(wy)
        score = consensus_ranking.loc[wy, 'consensus_score']
        rank = int(consensus_ranking.loc[wy, 'consensus_rank'])
        
        # Create hover text with detailed info
        hover_text = [
            f"<b>WY {wy}</b><br>"
            f"Date: {day_to_date_str(j)}<br>"
            f"SWE: {series[j]:.2f} in<br>"
            f"Consensus Rank: #{rank}<br>"
            f"Consensus Score: {score:.3f}"
            for j in range(len(series))
        ]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=series,
            mode='lines',
            name=f'WY {wy} (#{rank}, {score:.3f})',
            line=dict(color=colors['similar'][i % len(colors['similar'])], width=2),
            hovertext=hover_text,
            hoverinfo='text',
        ))
    
    # Add target year on top
    target_series = get_series(target_wy)
    hover_text_target = [
        f"<b>WY {target_wy} (TARGET)</b><br>"
        f"Date: {day_to_date_str(j)}<br>"
        f"SWE: {target_series[j]:.2f} in"
        for j in range(len(target_series))
    ]
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=target_series,
        mode='lines',
        name=f'WY {target_wy} (TARGET)',
        line=dict(color=colors['target'], width=3),
        hovertext=hover_text_target,
        hoverinfo='text',
    ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f'<b>Water Year {target_wy} vs. Most Similar Years (Consensus Ranking)</b><br>'
                 f'<span style="font-size:14px">{station_name} ({station_triplet})</span>',
            x=0.5,
            font=dict(size=18),
        ),
        xaxis=dict(
            title='Date',
            tickformat='%b',
            dtick='M1',
            tickangle=-45,
        ),
        yaxis=dict(
            title='Snow Water Equivalent (inches)',
            rangemode='tozero',
        ),
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
            bgcolor='rgba(255,255,255,0.8)',
        ),
        hovermode='x unified',
        template='plotly_white',
        height=600,
        width=1000,
    )
    
    return fig


def create_method_comparison_figure(
    wy_data: pd.DataFrame,
    target_wy: int,
    similarities: pd.DataFrame,
    station_name: str,
) -> go.Figure:
    """
    Create a multi-panel figure showing top years for each method.
    """
    
    methods = [
        ('dtw_distance', True, 'DTW (Timing Flexible)'),
        ('euclidean_distance', True, 'Euclidean (Point-by-Point)'),
        ('correlation', False, 'Correlation (Shape)'),
        ('lcss_similarity', False, 'LCSS (Gap Robust)'),
        ('edm_similarity', False, 'EDM (Dynamics)'),
    ]
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[m[2] for m in methods] + [''],
        specs=[[{}, {}], [{}, {}], [{}, {}]],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )
    
    # Date axis
    ref_year = 2000
    dates = [datetime(ref_year - 1, 10, 1) + timedelta(days=i) for i in range(365)]
    
    def get_series(wy):
        if wy not in wy_data.columns:
            return np.full(365, np.nan)
        series = wy_data[wy].values.copy()
        if len(series) == 366:
            series = normalize_to_365_days(series, is_leap_year=True)
        elif len(series) < 365:
            series = np.pad(series, (0, 365 - len(series)), constant_values=np.nan)
        series = impute_missing_linear(series[:365], max_gap=7)
        return series[:365]
    
    colors = ['#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']
    
    for idx, (col, ascending, title) in enumerate(methods):
        row = idx // 2 + 1
        col_idx = idx % 2 + 1
        
        if col not in similarities.columns:
            continue
        
        # Get top 4 similar years
        sorted_sim = similarities[col].sort_values(ascending=ascending)
        top_years = sorted_sim.head(4).index.tolist()
        
        # Plot similar years
        for i, wy in enumerate(top_years):
            series = get_series(wy)
            score = sorted_sim[wy]
            
            fig.add_trace(
                go.Scatter(
                    x=dates, y=series,
                    mode='lines',
                    name=f'WY {wy}',
                    line=dict(color=colors[i], width=1.5),
                    showlegend=(idx == 0),
                    legendgroup=f'wy{wy}',
                    hovertemplate=f'WY {wy}<br>%{{x|%b %d}}<br>SWE: %{{y:.2f}} in<extra></extra>',
                ),
                row=row, col=col_idx
            )
        
        # Plot target year
        target_series = get_series(target_wy)
        fig.add_trace(
            go.Scatter(
                x=dates, y=target_series,
                mode='lines',
                name=f'WY {target_wy} (target)',
                line=dict(color='#E63946', width=2.5),
                showlegend=(idx == 0),
                legendgroup='target',
                hovertemplate=f'WY {target_wy} (target)<br>%{{x|%b %d}}<br>SWE: %{{y:.2f}} in<extra></extra>',
            ),
            row=row, col=col_idx
        )
    
    fig.update_layout(
        title=dict(
            text=f'<b>Water Year {target_wy}: Most Similar Years by Method</b><br>'
                 f'<span style="font-size:14px">{station_name}</span>',
            x=0.5,
            font=dict(size=16),
        ),
        height=900,
        width=1200,
        template='plotly_white',
        hovermode='x unified',
    )
    
    # Update axes
    for i in range(1, 7):
        fig.update_xaxes(tickformat='%b', row=(i-1)//2 + 1, col=(i-1)%2 + 1)
        fig.update_yaxes(title_text='SWE (in)', rangemode='tozero', row=(i-1)//2 + 1, col=(i-1)%2 + 1)
    
    return fig


def create_consensus_ranking_table(
    consensus: pd.DataFrame,
    similarities: pd.DataFrame,
    n_show: int = 10,
) -> go.Figure:
    """Create a table showing the consensus ranking with method breakdown."""
    
    top_n = consensus.head(n_show)
    
    # Build table data
    headers = ['Rank', 'Water Year', 'Consensus<br>Score', 'DTW', 'Euclidean', 'Correlation', 'LCSS', 'EDM']
    
    cells = [
        [int(r) for r in top_n['consensus_rank']],
        [f'WY {wy}' for wy in top_n.index],
        [f'{s:.3f}' for s in top_n['consensus_score']],
    ]
    
    # Add individual method scores (normalized)
    for col in ['dtw_distance_normalized', 'euclidean_distance_normalized', 
                'correlation_normalized', 'lcss_similarity_normalized', 'edm_similarity_normalized']:
        if col in top_n.columns:
            cells.append([f'{s:.3f}' if pd.notna(s) else '-' for s in top_n[col]])
        else:
            cells.append(['-'] * len(top_n))
    
    # Color scale for cells (green = more similar)
    def score_to_color(score):
        if pd.isna(score):
            return 'white'
        # Green gradient
        r = int(255 - score * 100)
        g = int(200 + score * 55)
        b = int(200 - score * 100)
        return f'rgb({r},{g},{b})'
    
    fill_colors = [
        ['white'] * len(top_n),  # Rank
        ['white'] * len(top_n),  # Year
        [score_to_color(s) for s in top_n['consensus_score']],
    ]
    
    for col in ['dtw_distance_normalized', 'euclidean_distance_normalized',
                'correlation_normalized', 'lcss_similarity_normalized', 'edm_similarity_normalized']:
        if col in top_n.columns:
            fill_colors.append([score_to_color(s) for s in top_n[col]])
        else:
            fill_colors.append(['white'] * len(top_n))
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color='#457B9D',
            font=dict(color='white', size=12),
            align='center',
            height=35,
        ),
        cells=dict(
            values=cells,
            fill_color=fill_colors,
            align='center',
            font=dict(size=11),
            height=28,
        )
    )])
    
    fig.update_layout(
        title='<b>Consensus Ranking: Most Similar Water Years</b>',
        height=400,
        width=900,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    
    return fig


# =============================================================================
# MAIN VISUALIZATION FUNCTION
# =============================================================================

def generate_interactive_analysis(
    station_triplet: str,
    target_wy: int = 2020,
    output_dir: str = "data/processed",
    min_wy: int = 1996,
    max_wy: int = 2025,
) -> dict:
    """
    Generate complete interactive analysis with consensus ranking.
    
    Returns dict with paths to generated files.
    """
    print(f"\n{'='*60}")
    print(f"Interactive Water Year Similarity Analysis")
    print(f"{'='*60}")
    print(f"Station: {station_triplet}")
    print(f"Target: WY {target_wy}")
    print()
    
    engine = create_engine(DATABASE_URL)
    
    # Get station info
    stations = get_available_stations(engine)
    station_info = stations[stations['triplet'] == station_triplet]
    station_name = station_info.iloc[0]['name'] if not station_info.empty else station_triplet
    
    # Extract data
    print("Extracting water year data...")
    wy_data = extract_station_water_years(engine, station_triplet, min_wy, max_wy)
    
    if wy_data.empty:
        print("ERROR: No data found")
        return None
    
    # Filter valid years
    coverage = wy_data.notna().sum() / len(wy_data)
    valid_wys = coverage[coverage > 0.7].index.tolist()
    wy_data = wy_data[valid_wys]
    
    print(f"Found {len(wy_data.columns)} water years with sufficient data")
    
    # Compute similarities
    print("Computing similarities using all methods...")
    similarities = compute_all_similarities(
        wy_data, target_wy,
        methods=["dtw", "lcss", "correlation", "euclidean", "edm"]
    )
    
    # Compute consensus ranking
    print("Computing consensus ranking...")
    consensus = compute_consensus_ranking(similarities)
    
    # Print top 10
    print(f"\n{'='*60}")
    print(f"Top 10 Most Similar Water Years to WY {target_wy}")
    print(f"{'='*60}")
    for i, (wy, row) in enumerate(consensus.head(10).iterrows()):
        print(f"  #{i+1}: WY {wy} (consensus score: {row['consensus_score']:.3f})")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_filename = f"wy_similarity_{station_triplet.replace(':', '_')}_wy{target_wy}"
    
    # Generate figures
    print("Generating interactive visualizations...")
    
    # 1. Consensus comparison plot
    fig_consensus = create_interactive_comparison(
        wy_data, target_wy, consensus, station_name, station_triplet
    )
    consensus_path = output_path / f"{base_filename}_consensus.html"
    fig_consensus.write_html(str(consensus_path))
    print(f"  Saved: {consensus_path}")
    
    # 2. Method comparison plot
    fig_methods = create_method_comparison_figure(
        wy_data, target_wy, similarities, station_name
    )
    methods_path = output_path / f"{base_filename}_methods.html"
    fig_methods.write_html(str(methods_path))
    print(f"  Saved: {methods_path}")
    
    # 3. Ranking table
    fig_table = create_consensus_ranking_table(consensus, similarities)
    table_path = output_path / f"{base_filename}_ranking.html"
    fig_table.write_html(str(table_path))
    print(f"  Saved: {table_path}")
    
    # 4. Combined dashboard HTML
    dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Water Year {target_wy} Similarity Analysis - {station_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #457B9D 0%, #1D3557 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #1D3557;
            border-bottom: 2px solid #E63946;
            padding-bottom: 10px;
        }}
        iframe {{
            width: 100%;
            border: none;
            border-radius: 5px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 1200px) {{
            .grid {{ grid-template-columns: 1fr; }}
        }}
        .method-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #457B9D;
        }}
        .method-card h3 {{
            margin-top: 0;
            color: #1D3557;
        }}
        .top-years {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .year-badge {{
            background: #457B9D;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 14px;
        }}
        .year-badge.target {{
            background: #E63946;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Water Year {target_wy} Similarity Analysis</h1>
        <p>{station_name} ({station_triplet})</p>
    </div>
    
    <div class="section">
        <h2>🏆 Consensus Ranking</h2>
        <p>The consensus ranking combines all five similarity methods (DTW, Euclidean, Correlation, LCSS, EDM) into a single score. Each method is normalized to 0-1 scale and averaged.</p>
        <iframe src="{base_filename}_ranking.html" height="420"></iframe>
    </div>
    
    <div class="section">
        <h2>📊 Interactive Comparison (Consensus Top 5)</h2>
        <p>Hover over the lines to see exact SWE values. Click legend items to toggle years on/off.</p>
        <iframe src="{base_filename}_consensus.html" height="650"></iframe>
    </div>
    
    <div class="section">
        <h2>🔍 Method-by-Method Comparison</h2>
        <p>Each panel shows the top 4 similar years according to that specific method.</p>
        <iframe src="{base_filename}_methods.html" height="950"></iframe>
    </div>
    
    <div class="section">
        <h2>📖 Method Descriptions</h2>
        <div class="grid">
            <div class="method-card">
                <h3>Dynamic Time Warping (DTW)</h3>
                <p>Finds optimal alignment between time series, allowing for temporal stretching. Best for years with similar patterns but different timing (e.g., early vs late melt).</p>
            </div>
            <div class="method-card">
                <h3>Euclidean Distance</h3>
                <p>Point-by-point comparison of SWE values. Best for finding years with nearly identical daily values.</p>
            </div>
            <div class="method-card">
                <h3>Pearson Correlation</h3>
                <p>Measures shape similarity regardless of magnitude. A low-snow year can match a high-snow year if they have the same seasonal pattern.</p>
            </div>
            <div class="method-card">
                <h3>LCSS (Longest Common Subsequence)</h3>
                <p>Robust to missing data and noise. Finds the longest matching subsequence within a tolerance threshold.</p>
            </div>
            <div class="method-card">
                <h3>EDM (State-Space Similarity)</h3>
                <p>Based on Empirical Dynamic Modeling. Compares trajectories in reconstructed state space to find years with similar dynamics.</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    dashboard_path = output_path / f"{base_filename}_dashboard.html"
    with open(dashboard_path, 'w') as f:
        f.write(dashboard_html)
    print(f"  Saved: {dashboard_path}")
    
    print(f"\n{'='*60}")
    print(f"✓ Analysis complete!")
    print(f"{'='*60}")
    print(f"\nOpen the dashboard in your browser:")
    print(f"  {dashboard_path}")
    
    return {
        'consensus': str(consensus_path),
        'methods': str(methods_path),
        'table': str(table_path),
        'dashboard': str(dashboard_path),
        'consensus_ranking': consensus,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Water Year Similarity Analysis")
    parser.add_argument("--station", type=str, default="473:CA:SNTL",
                       help="Station triplet")
    parser.add_argument("--target-wy", type=int, default=2020,
                       help="Target water year")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                       help="Output directory")
    
    args = parser.parse_args()
    
    generate_interactive_analysis(
        args.station,
        args.target_wy,
        args.output_dir,
    )

