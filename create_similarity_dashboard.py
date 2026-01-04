"""
Create Interactive Water Year Similarity Dashboard

Generates an HTML dashboard with:
- Station selector dropdown
- Method selector (Correlation, DTW, FFT, EDM)
- Raw vs Delta toggle
- Interactive heatmap matrix
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform, cosine
from sqlalchemy import create_engine

from config import DATABASE_URL
from water_year_similarity import (
    extract_station_water_years,
    get_available_stations,
    correlation_similarity,
    delta_correlation,
    dtw_distance,
    delta_dtw,
    edm_simplex_similarity,
    normalize_to_365_days,
    impute_missing_linear,
    prepare_series_pair,
    compute_delta,
)


def get_clean_series(wy_data: pd.DataFrame, wy: int) -> np.ndarray:
    """Get a clean 365-day series for a water year."""
    series = wy_data[wy].values.copy()
    if len(series) == 366:
        series = normalize_to_365_days(series, is_leap_year=True)
    elif len(series) < 365:
        series = np.pad(series, (0, 365 - len(series)), constant_values=np.nan)
    return impute_missing_linear(series[:365], max_gap=14)


def compute_fft_similarity(s1: np.ndarray, s2: np.ndarray, n_coef: int = 20) -> float:
    """Compute FFT-based similarity."""
    fft1 = np.fft.fft(s1 - np.mean(s1))[:n_coef]
    fft2 = np.fft.fft(s2 - np.mean(s2))[:n_coef]
    mag1 = np.abs(fft1)
    mag2 = np.abs(fft2)
    if np.sum(mag1) == 0 or np.sum(mag2) == 0:
        return np.nan
    return 1 - cosine(mag1, mag2)


def compute_delta_fft_similarity(s1: np.ndarray, s2: np.ndarray, n_coef: int = 20) -> float:
    """Compute FFT-based similarity on delta series."""
    d1 = compute_delta(s1)
    d2 = compute_delta(s2)
    return compute_fft_similarity(d1, d2, n_coef)


def compute_pairwise_matrix(
    wy_data: pd.DataFrame,
    method: str = "correlation",
    use_delta: bool = False,
) -> pd.DataFrame:
    """Compute pairwise similarity/distance matrix."""
    valid_wys = sorted(wy_data.columns.tolist())
    n = len(valid_wys)
    matrix = np.zeros((n, n))
    
    # Get clean series for all years
    series_dict = {wy: get_clean_series(wy_data, wy) for wy in valid_wys}
    
    for i, wy1 in enumerate(valid_wys):
        s1 = series_dict[wy1]
        for j, wy2 in enumerate(valid_wys):
            if i == j:
                # Diagonal: max similarity or 0 distance
                if method in ["dtw"]:
                    matrix[i, j] = 0
                else:
                    matrix[i, j] = 1.0
            elif j > i:
                s2 = series_dict[wy2]
                
                # Compute similarity based on method
                if method == "correlation":
                    if use_delta:
                        val = delta_correlation(s1, s2)
                    else:
                        val = correlation_similarity(s1, s2)
                elif method == "dtw":
                    if use_delta:
                        val = delta_dtw(s1, s2)
                    else:
                        val = dtw_distance(s1, s2)
                elif method == "fft":
                    if use_delta:
                        val = compute_delta_fft_similarity(s1, s2)
                    else:
                        val = compute_fft_similarity(s1, s2)
                elif method == "edm":
                    if use_delta:
                        # EDM on delta
                        d1, d2 = compute_delta(s1), compute_delta(s2)
                        val = edm_simplex_similarity(d1, d2)
                    else:
                        val = edm_simplex_similarity(s1, s2)
                else:
                    val = np.nan
                
                matrix[i, j] = val
                matrix[j, i] = val
    
    return pd.DataFrame(matrix, index=valid_wys, columns=valid_wys)


def get_cluster_order(matrix_df: pd.DataFrame, is_distance: bool = False) -> list:
    """Get optimal ordering via hierarchical clustering."""
    values = matrix_df.values.copy()
    
    if is_distance:
        dist = values.copy()
    else:
        # Convert similarity to distance
        dist = 1 - values
    
    dist = np.clip(dist, 0, 2)
    np.fill_diagonal(dist, 0)
    
    try:
        link = linkage(squareform(dist), method='ward')
        order = leaves_list(link)
        return [matrix_df.index[i] for i in order]
    except Exception:
        return list(matrix_df.index)


def generate_dashboard_html(station_data: dict, output_path: str):
    """Generate the interactive HTML dashboard."""
    
    # Prepare data for JavaScript
    js_data = json.dumps(station_data)
    
    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Water Year Similarity Matrix Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
            font-size: 2rem;
            color: #4fc3f7;
            text-shadow: 0 0 20px rgba(79, 195, 247, 0.3);
        }}
        .subtitle {{
            text-align: center;
            color: #90a4ae;
            margin-bottom: 25px;
            font-size: 0.95rem;
        }}
        .controls {{
            display: flex;
            gap: 20px;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 25px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .control-group label {{
            font-size: 0.85rem;
            color: #90a4ae;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        select {{
            padding: 10px 15px;
            font-size: 1rem;
            border: 1px solid #37474f;
            border-radius: 8px;
            background: #263238;
            color: #e0e0e0;
            cursor: pointer;
            min-width: 200px;
            transition: all 0.2s;
        }}
        select:hover {{
            border-color: #4fc3f7;
        }}
        select:focus {{
            outline: none;
            border-color: #4fc3f7;
            box-shadow: 0 0 0 3px rgba(79, 195, 247, 0.2);
        }}
        .toggle-group {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .toggle-btn {{
            padding: 10px 20px;
            border: 1px solid #37474f;
            border-radius: 8px;
            background: #263238;
            color: #90a4ae;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.95rem;
        }}
        .toggle-btn:hover {{
            border-color: #4fc3f7;
        }}
        .toggle-btn.active {{
            background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%);
            color: #1a1a2e;
            border-color: transparent;
            font-weight: 600;
        }}
        #heatmap {{
            background: rgba(255, 255, 255, 0.02);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .stat-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 18px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .stat-card h3 {{
            font-size: 0.85rem;
            color: #90a4ae;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .stat-value {{
            font-size: 1.1rem;
            color: #4fc3f7;
        }}
        .stat-detail {{
            font-size: 0.9rem;
            color: #b0bec5;
            margin-top: 4px;
        }}
        .loading {{
            text-align: center;
            padding: 60px;
            color: #90a4ae;
        }}
        .method-info {{
            background: rgba(79, 195, 247, 0.1);
            border-left: 3px solid #4fc3f7;
            padding: 12px 15px;
            margin-bottom: 20px;
            border-radius: 0 8px 8px 0;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏔️ Water Year Similarity Matrix</h1>
        <p class="subtitle">Compare similarity between water years across SNOTEL stations</p>
        
        <div class="controls">
            <div class="control-group">
                <label>Station</label>
                <select id="stationSelect"></select>
            </div>
            <div class="control-group">
                <label>Method</label>
                <select id="methodSelect">
                    <option value="correlation">Correlation</option>
                    <option value="dtw">Dynamic Time Warping (DTW)</option>
                    <option value="fft">FFT Spectral</option>
                    <option value="edm">Empirical Dynamic Modeling</option>
                </select>
            </div>
            <div class="control-group">
                <label>Data Type</label>
                <div class="toggle-group">
                    <button class="toggle-btn active" data-value="raw" onclick="setDataType('raw')">Raw SWE</button>
                    <button class="toggle-btn" data-value="delta" onclick="setDataType('delta')">Δ SWE</button>
                </div>
            </div>
        </div>
        
        <div class="method-info" id="methodInfo"></div>
        
        <div id="heatmap">
            <div class="loading">Loading data...</div>
        </div>
        
        <div class="stats" id="stats"></div>
    </div>
    
    <script>
        // Data from Python
        const stationData = {js_data};
        
        let currentStation = null;
        let currentMethod = 'correlation';
        let currentDataType = 'raw';
        
        const methodDescriptions = {{
            correlation: "Pearson correlation measures linear relationship between SWE patterns. Values range from -1 to 1, where 1 = identical shape.",
            dtw: "Dynamic Time Warping finds optimal alignment between series, handling timing shifts. Lower distance = more similar.",
            fft: "FFT compares frequency content (spectral signature). Captures periodic patterns regardless of phase. Higher = more similar.",
            edm: "Empirical Dynamic Modeling measures state-space trajectory similarity. Based on Sugihara et al. Higher = more similar dynamics."
        }};
        
        // Initialize station dropdown
        function initStationDropdown() {{
            const select = document.getElementById('stationSelect');
            const stations = Object.keys(stationData).sort((a, b) => {{
                return stationData[a].name.localeCompare(stationData[b].name);
            }});
            
            stations.forEach(triplet => {{
                const opt = document.createElement('option');
                opt.value = triplet;
                opt.textContent = `${{stationData[triplet].name}} (${{stationData[triplet].state}})`;
                select.appendChild(opt);
            }});
            
            currentStation = stations[0];
            select.value = currentStation;
        }}
        
        function setDataType(type) {{
            currentDataType = type;
            document.querySelectorAll('.toggle-btn').forEach(btn => {{
                btn.classList.toggle('active', btn.dataset.value === type);
            }});
            updatePlot();
        }}
        
        function updateMethodInfo() {{
            const delta = currentDataType === 'delta' ? ' (applied to daily changes)' : '';
            document.getElementById('methodInfo').innerHTML = methodDescriptions[currentMethod] + delta;
        }}
        
        function updatePlot() {{
            if (!currentStation) return;
            
            updateMethodInfo();
            
            const key = `${{currentMethod}}_${{currentDataType}}`;
            const data = stationData[currentStation];
            
            if (!data.matrices[key]) {{
                document.getElementById('heatmap').innerHTML = '<div class="loading">Data not available for this combination</div>';
                return;
            }}
            
            const matrix = data.matrices[key];
            const years = data.years;
            const order = data.orders[key] || years;
            
            // Reorder matrix by cluster order
            const orderedMatrix = order.map(y1 => order.map(y2 => matrix[years.indexOf(y1)][years.indexOf(y2)]));
            
            // Determine color scale
            const isDistance = currentMethod === 'dtw';
            let colorscale, zmin, zmax, colorbarTitle;
            
            if (isDistance) {{
                colorscale = 'YlOrRd';
                zmin = 0;
                zmax = Math.max(...orderedMatrix.flat());
                colorbarTitle = 'Distance';
            }} else {{
                colorscale = 'RdBu_r';
                zmin = -0.3;
                zmax = 1;
                colorbarTitle = 'Similarity';
            }}
            
            // Create hover text
            const hovertext = order.map((y1, i) => 
                order.map((y2, j) => {{
                    const val = orderedMatrix[i][j];
                    const label = isDistance ? 'Distance' : 'Similarity';
                    return `WY ${{y1}} vs WY ${{y2}}<br>${{label}}: ${{val.toFixed(3)}}`;
                }})
            );
            
            const trace = {{
                z: orderedMatrix,
                x: order,
                y: order,
                type: 'heatmap',
                colorscale: colorscale,
                zmin: zmin,
                zmax: zmax,
                text: hovertext,
                hoverinfo: 'text',
                colorbar: {{
                    title: colorbarTitle,
                    titleside: 'right'
                }}
            }};
            
            const layout = {{
                title: {{
                    text: `${{data.name}} - ${{currentMethod.toUpperCase()}} (${{currentDataType === 'delta' ? 'Δ SWE' : 'Raw SWE'}})`,
                    font: {{ size: 16, color: '#e0e0e0' }}
                }},
                xaxis: {{
                    tickangle: 45,
                    tickfont: {{ size: 10, color: '#b0bec5' }},
                    gridcolor: 'rgba(255,255,255,0.1)'
                }},
                yaxis: {{
                    tickfont: {{ size: 10, color: '#b0bec5' }},
                    autorange: 'reversed',
                    gridcolor: 'rgba(255,255,255,0.1)'
                }},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: {{ t: 80, r: 100, b: 80, l: 60 }},
                height: 650
            }};
            
            Plotly.newPlot('heatmap', [trace], layout, {{responsive: true}});
            
            // Update stats
            updateStats(matrix, years, isDistance);
        }}
        
        function updateStats(matrix, years, isDistance) {{
            const n = years.length;
            let maxVal = isDistance ? Infinity : -Infinity;
            let minVal = isDistance ? -Infinity : Infinity;
            let maxPair = [], minPair = [];
            let sum = 0, count = 0;
            
            for (let i = 0; i < n; i++) {{
                for (let j = i + 1; j < n; j++) {{
                    const val = matrix[i][j];
                    if (isNaN(val)) continue;
                    
                    sum += val;
                    count++;
                    
                    if (isDistance) {{
                        if (val < maxVal) {{ maxVal = val; maxPair = [years[i], years[j]]; }}
                        if (val > minVal) {{ minVal = val; minPair = [years[i], years[j]]; }}
                    }} else {{
                        if (val > maxVal) {{ maxVal = val; maxPair = [years[i], years[j]]; }}
                        if (val < minVal) {{ minVal = val; minPair = [years[i], years[j]]; }}
                    }}
                }}
            }}
            
            const avg = sum / count;
            
            // Year averages
            const yearAvgs = years.map((y, i) => {{
                let ySum = 0, yCount = 0;
                for (let j = 0; j < n; j++) {{
                    if (i !== j && !isNaN(matrix[i][j])) {{
                        ySum += matrix[i][j];
                        yCount++;
                    }}
                }}
                return {{ year: y, avg: ySum / yCount }};
            }}).sort((a, b) => isDistance ? a.avg - b.avg : b.avg - a.avg);
            
            const typical = yearAvgs[0];
            const unusual = yearAvgs[yearAvgs.length - 1];
            
            const label = isDistance ? 'distance' : 'similarity';
            const mostLabel = isDistance ? 'Most Similar (lowest distance)' : 'Most Similar';
            const leastLabel = isDistance ? 'Least Similar (highest distance)' : 'Least Similar';
            
            document.getElementById('stats').innerHTML = `
                <div class="stat-card">
                    <h3>${{mostLabel}}</h3>
                    <div class="stat-value">WY ${{maxPair[0]}} & WY ${{maxPair[1]}}</div>
                    <div class="stat-detail">${{label}}: ${{maxVal.toFixed(3)}}</div>
                </div>
                <div class="stat-card">
                    <h3>${{leastLabel}}</h3>
                    <div class="stat-value">WY ${{minPair[0]}} & WY ${{minPair[1]}}</div>
                    <div class="stat-detail">${{label}}: ${{minVal.toFixed(3)}}</div>
                </div>
                <div class="stat-card">
                    <h3>Most Typical Year</h3>
                    <div class="stat-value">WY ${{typical.year}}</div>
                    <div class="stat-detail">avg ${{label}}: ${{typical.avg.toFixed(3)}}</div>
                </div>
                <div class="stat-card">
                    <h3>Most Unusual Year</h3>
                    <div class="stat-value">WY ${{unusual.year}}</div>
                    <div class="stat-detail">avg ${{label}}: ${{unusual.avg.toFixed(3)}}</div>
                </div>
            `;
        }}
        
        // Event listeners
        document.getElementById('stationSelect').addEventListener('change', (e) => {{
            currentStation = e.target.value;
            updatePlot();
        }});
        
        document.getElementById('methodSelect').addEventListener('change', (e) => {{
            currentMethod = e.target.value;
            updatePlot();
        }});
        
        // Initialize
        initStationDropdown();
        updatePlot();
    </script>
</body>
</html>'''
    
    with open(output_path, 'w') as f:
        f.write(html_template)
    
    print(f"Dashboard saved to: {output_path}")


def main():
    """Main function to generate the dashboard."""
    engine = create_engine(DATABASE_URL)
    
    # Get top stations by observation count
    stations_df = get_available_stations(engine)
    top_stations = stations_df.head(15)  # Top 15 stations
    
    print(f"Processing {len(top_stations)} stations...")
    
    methods = ["correlation", "dtw", "fft", "edm"]
    data_types = ["raw", "delta"]
    
    station_data = {}
    
    for idx, row in top_stations.iterrows():
        triplet = row['triplet']
        print(f"\n  Processing: {row['name']} ({triplet})")
        
        # Get water year data
        wy_data = extract_station_water_years(engine, triplet, 1996, 2025)
        coverage = wy_data.notna().sum() / len(wy_data)
        valid_wys = sorted(coverage[coverage > 0.7].index.tolist())
        
        if len(valid_wys) < 5:
            print(f"    Skipping: only {len(valid_wys)} valid years")
            continue
        
        wy_data = wy_data[valid_wys]
        
        station_info = {
            "name": row['name'],
            "state": row['state_code'],
            "years": [int(y) for y in valid_wys],  # Convert to Python int for JSON
            "matrices": {},
            "orders": {}
        }
        
        for method in methods:
            for data_type in data_types:
                key = f"{method}_{data_type}"
                use_delta = data_type == "delta"
                
                print(f"    Computing: {key}")
                
                try:
                    matrix_df = compute_pairwise_matrix(wy_data, method, use_delta)
                    
                    # Get cluster order
                    is_distance = method == "dtw"
                    order = get_cluster_order(matrix_df, is_distance)
                    
                    # Store as list of lists for JSON (convert numpy types)
                    station_info["matrices"][key] = [[float(v) for v in row] for row in matrix_df.values]
                    station_info["orders"][key] = [int(y) for y in order]
                except Exception as e:
                    print(f"      Error: {e}")
        
        station_data[triplet] = station_info
    
    # Generate HTML
    output_path = Path("data/processed/wy_similarity_dashboard.html")
    generate_dashboard_html(station_data, str(output_path))
    
    print(f"\n✓ Dashboard created: {output_path}")
    print(f"  Stations included: {len(station_data)}")


if __name__ == "__main__":
    main()

