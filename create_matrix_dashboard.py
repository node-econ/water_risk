"""
Create Simple Interactive Water Year Similarity Matrix Dashboard

Generates a clean HTML dashboard with:
- Station selector dropdown
- Method selector (Correlation, DTW, FFT, EDM)  
- Raw vs Delta toggle
- Clear n×n matrix heatmap with values
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cosine
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
                if method == "dtw":
                    matrix[i, j] = 0
                else:
                    matrix[i, j] = 1.0
            elif j > i:
                s2 = series_dict[wy2]
                
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
                        d1, d2 = compute_delta(s1), compute_delta(s2)
                        val = compute_fft_similarity(d1, d2)
                    else:
                        val = compute_fft_similarity(s1, s2)
                elif method == "edm":
                    if use_delta:
                        d1, d2 = compute_delta(s1), compute_delta(s2)
                        val = edm_simplex_similarity(d1, d2)
                    else:
                        val = edm_simplex_similarity(s1, s2)
                else:
                    val = np.nan
                
                matrix[i, j] = val
                matrix[j, i] = val
    
    return pd.DataFrame(matrix, index=valid_wys, columns=valid_wys)


def generate_dashboard(output_path: str):
    """Generate the interactive HTML dashboard."""
    
    engine = create_engine(DATABASE_URL)
    
    # Get top stations
    stations_df = get_available_stations(engine)
    top_stations = stations_df.head(10)
    
    print(f"Processing {len(top_stations)} stations...")
    
    methods = ["correlation", "dtw", "fft", "edm"]
    data_types = ["raw", "delta"]
    
    # Build JavaScript data object
    js_stations = []
    js_matrices = {}
    
    for idx, row in top_stations.iterrows():
        triplet = row['triplet']
        name = row['name']
        state = row['state_code']
        
        print(f"  Processing: {name}")
        
        # Get water year data
        wy_data = extract_station_water_years(engine, triplet, 1996, 2025)
        coverage = wy_data.notna().sum() / len(wy_data)
        valid_wys = sorted(coverage[coverage > 0.7].index.tolist())
        
        if len(valid_wys) < 5:
            continue
        
        wy_data = wy_data[valid_wys]
        years = [int(y) for y in valid_wys]
        
        js_stations.append({
            "triplet": triplet,
            "name": name,
            "state": state,
            "years": years
        })
        
        for method in methods:
            for dtype in data_types:
                key = f"{triplet}_{method}_{dtype}"
                use_delta = dtype == "delta"
                
                try:
                    matrix_df = compute_pairwise_matrix(wy_data, method, use_delta)
                    # Convert to list of lists with proper Python floats
                    matrix_list = []
                    for i in range(len(years)):
                        row = []
                        for j in range(len(years)):
                            val = float(matrix_df.iloc[i, j])
                            if np.isnan(val):
                                row.append(None)
                            else:
                                row.append(round(val, 3))
                        matrix_list.append(row)
                    js_matrices[key] = matrix_list
                except Exception as e:
                    print(f"    Error computing {key}: {e}")
                    js_matrices[key] = None
    
    # Generate HTML
    import json
    stations_json = json.dumps(js_stations)
    matrices_json = json.dumps(js_matrices)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Water Year Similarity Matrix</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            font-size: 1.8rem;
            margin-bottom: 8px;
            color: #38bdf8;
        }}
        .subtitle {{
            text-align: center;
            color: #94a3b8;
            margin-bottom: 24px;
        }}
        .controls {{
            display: flex;
            gap: 16px;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 20px;
            padding: 16px;
            background: #1e293b;
            border-radius: 8px;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .control-group label {{
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
        }}
        select {{
            padding: 8px 12px;
            font-size: 0.95rem;
            border: 1px solid #334155;
            border-radius: 6px;
            background: #0f172a;
            color: #e2e8f0;
            min-width: 180px;
        }}
        .toggle-group {{ display: flex; gap: 8px; }}
        .toggle-btn {{
            padding: 8px 16px;
            border: 1px solid #334155;
            border-radius: 6px;
            background: #0f172a;
            color: #94a3b8;
            cursor: pointer;
            font-size: 0.9rem;
        }}
        .toggle-btn.active {{
            background: #38bdf8;
            color: #0f172a;
            border-color: #38bdf8;
            font-weight: 600;
        }}
        #chart {{ 
            background: #1e293b; 
            border-radius: 8px; 
            padding: 16px;
            min-height: 700px;
        }}
        .info {{
            background: #1e293b;
            border-left: 3px solid #38bdf8;
            padding: 12px;
            margin-bottom: 16px;
            border-radius: 0 6px 6px 0;
            font-size: 0.9rem;
            color: #94a3b8;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-top: 16px;
        }}
        .stat-card {{
            background: #1e293b;
            padding: 12px;
            border-radius: 6px;
        }}
        .stat-card h3 {{
            font-size: 0.7rem;
            color: #64748b;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        .stat-value {{ color: #38bdf8; font-size: 1rem; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Water Year Similarity Matrix</h1>
        <p class="subtitle">N×N correlation matrix comparing water years</p>
        
        <div class="controls">
            <div class="control-group">
                <label>Station</label>
                <select id="station"></select>
            </div>
            <div class="control-group">
                <label>Method</label>
                <select id="method">
                    <option value="correlation">Correlation</option>
                    <option value="dtw">DTW Distance</option>
                    <option value="fft">FFT Spectral</option>
                    <option value="edm">EDM</option>
                </select>
            </div>
            <div class="control-group">
                <label>Data</label>
                <div class="toggle-group">
                    <button class="toggle-btn active" id="btn-raw">Raw SWE</button>
                    <button class="toggle-btn" id="btn-delta">Δ SWE</button>
                </div>
            </div>
        </div>
        
        <div class="info" id="info">Select options to view the similarity matrix.</div>
        
        <div id="chart"></div>
        
        <div class="stats" id="stats"></div>
    </div>

    <script>
        const stations = {stations_json};
        const matrices = {matrices_json};
        
        let currentStation = null;
        let currentMethod = 'correlation';
        let currentData = 'raw';
        
        const methodInfo = {{
            correlation: 'Pearson correlation: 1.0 = identical, 0 = no relationship, -1 = inverse',
            dtw: 'Dynamic Time Warping distance: 0 = identical, higher = more different',
            fft: 'FFT spectral similarity: 1.0 = identical frequency content, 0 = different',
            edm: 'Empirical Dynamic Modeling: 1.0 = identical dynamics, 0 = different'
        }};
        
        // Initialize station dropdown
        const stationSelect = document.getElementById('station');
        stations.forEach(s => {{
            const opt = document.createElement('option');
            opt.value = s.triplet;
            opt.textContent = s.name + ' (' + s.state + ')';
            stationSelect.appendChild(opt);
        }});
        currentStation = stations[0];
        
        // Event listeners
        stationSelect.addEventListener('change', e => {{
            currentStation = stations.find(s => s.triplet === e.target.value);
            updateChart();
        }});
        
        document.getElementById('method').addEventListener('change', e => {{
            currentMethod = e.target.value;
            updateChart();
        }});
        
        document.getElementById('btn-raw').addEventListener('click', () => {{
            currentData = 'raw';
            document.getElementById('btn-raw').classList.add('active');
            document.getElementById('btn-delta').classList.remove('active');
            updateChart();
        }});
        
        document.getElementById('btn-delta').addEventListener('click', () => {{
            currentData = 'delta';
            document.getElementById('btn-delta').classList.add('active');
            document.getElementById('btn-raw').classList.remove('active');
            updateChart();
        }});
        
        function updateChart() {{
            const key = currentStation.triplet + '_' + currentMethod + '_' + currentData;
            const matrix = matrices[key];
            const years = currentStation.years;
            
            document.getElementById('info').textContent = methodInfo[currentMethod] + 
                (currentData === 'delta' ? ' (applied to daily changes)' : '');
            
            if (!matrix) {{
                document.getElementById('chart').innerHTML = '<p style="padding:40px;text-align:center;">Data not available</p>';
                return;
            }}
            
            const isDistance = currentMethod === 'dtw';
            
            // Create a copy of matrix with diagonal set to null (will show as white)
            const displayMatrix = matrix.map((row, i) => 
                row.map((val, j) => i === j ? null : val)
            );
            
            // Create hover text with values
            const hoverText = years.map((y1, i) => 
                years.map((y2, j) => {{
                    if (i === j) return 'WY ' + y1 + ' (self)';
                    const val = matrix[i][j];
                    return 'WY ' + y1 + ' vs WY ' + y2 + '<br>Value: ' + (val !== null ? val.toFixed(3) : 'N/A');
                }})
            );
            
            // Create annotation text (show values in cells)
            const annotations = [];
            for (let i = 0; i < years.length; i++) {{
                for (let j = 0; j < years.length; j++) {{
                    const val = matrix[i][j];
                    if (val !== null) {{
                        const isDiagonal = i === j;
                        annotations.push({{
                            x: years[j],
                            y: years[i],
                            text: isDiagonal ? '' : val.toFixed(2),  // Hide text on diagonal
                            showarrow: false,
                            font: {{ size: 8, color: isDiagonal ? 'white' : (Math.abs(val) > 0.5 || (isDistance && val < 20) ? 'white' : '#666') }}
                        }});
                    }}
                }}
            }}
            
            const trace = {{
                z: displayMatrix,
                x: years,
                y: years,
                type: 'heatmap',
                colorscale: isDistance ? 'YlOrRd' : 'RdBu_r',
                zmin: isDistance ? 0 : -0.3,
                zmax: isDistance ? Math.max(...matrix.flat().filter(v => v !== null)) : 1,
                text: hoverText,
                hoverinfo: 'text',
                colorbar: {{ title: isDistance ? 'Distance' : 'Similarity' }}
            }};
            
            const layout = {{
                title: {{
                    text: currentStation.name + ' - ' + currentMethod.toUpperCase() + ' (' + (currentData === 'delta' ? 'Δ SWE' : 'Raw SWE') + ')',
                    font: {{ color: '#e2e8f0' }}
                }},
                xaxis: {{
                    title: 'Water Year',
                    tickfont: {{ size: 10, color: '#94a3b8' }},
                    tickangle: -45,
                    gridcolor: '#334155'
                }},
                yaxis: {{
                    title: 'Water Year',
                    tickfont: {{ size: 10, color: '#94a3b8' }},
                    autorange: 'reversed',
                    gridcolor: '#334155'
                }},
                annotations: annotations,
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                height: 700,
                margin: {{ l: 60, r: 80, t: 60, b: 60 }}
            }};
            
            Plotly.newPlot('chart', [trace], layout, {{responsive: true}});
            
            // Update stats
            updateStats(matrix, years, isDistance);
        }}
        
        function updateStats(matrix, years, isDistance) {{
            let best = {{ val: isDistance ? Infinity : -Infinity, pair: [] }};
            let worst = {{ val: isDistance ? -Infinity : Infinity, pair: [] }};
            
            for (let i = 0; i < years.length; i++) {{
                for (let j = i + 1; j < years.length; j++) {{
                    const v = matrix[i][j];
                    if (v === null) continue;
                    if (isDistance) {{
                        if (v < best.val) {{ best = {{ val: v, pair: [years[i], years[j]] }}; }}
                        if (v > worst.val) {{ worst = {{ val: v, pair: [years[i], years[j]] }}; }}
                    }} else {{
                        if (v > best.val) {{ best = {{ val: v, pair: [years[i], years[j]] }}; }}
                        if (v < worst.val) {{ worst = {{ val: v, pair: [years[i], years[j]] }}; }}
                    }}
                }}
            }}
            
            document.getElementById('stats').innerHTML = `
                <div class="stat-card">
                    <h3>${{isDistance ? 'Most Similar (Min Distance)' : 'Most Similar'}}</h3>
                    <div class="stat-value">WY ${{best.pair[0]}} & WY ${{best.pair[1]}}</div>
                    <div style="color:#64748b;font-size:0.85rem">${{best.val.toFixed(3)}}</div>
                </div>
                <div class="stat-card">
                    <h3>${{isDistance ? 'Least Similar (Max Distance)' : 'Least Similar'}}</h3>
                    <div class="stat-value">WY ${{worst.pair[0]}} & WY ${{worst.pair[1]}}</div>
                    <div style="color:#64748b;font-size:0.85rem">${{worst.val.toFixed(3)}}</div>
                </div>
            `;
        }}
        
        // Initial render
        updateChart();
    </script>
</body>
</html>'''
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"\n✓ Dashboard saved: {output_path}")


if __name__ == "__main__":
    generate_dashboard("data/processed/wy_matrix_dashboard.html")

