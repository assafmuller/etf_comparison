
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, date

# Helper functions
def parse_date(s):
    if isinstance(s, date):
        return s
    for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Date {s} is not in a recognized format.")

def calculate_roi(prices, start_date):
    try:
        if not prices:
            return ''
        # Require at least one price on or before start_date
        dates = [parse_date(d) for d, _ in prices]
        if not any(d <= start_date for d in dates):
            return ''
        # Use the closest price ON or AFTER start_date
        start_candidates = [(parse_date(d), v) for d, v in prices if parse_date(d) >= start_date]
        _, start_price = min(start_candidates, key=lambda x: x[0])
        # Use the last available price as the end price
        _, end_price = max((parse_date(d), v) for d, v in prices)
        return (end_price - start_price) / start_price
    except Exception:
        return ''

def sharpe_ratio(returns, risk_free=0):
    if len(returns) < 2:
        return ''
    excess = np.array(returns) - risk_free
    return np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(52)

def sortino_ratio(returns, risk_free=0):
    if len(returns) < 2:
        return ''
    excess = np.array(returns) - risk_free
    downside = excess[excess < 0]
    denom = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 0
    return np.mean(excess) / denom * np.sqrt(52) if denom > 0 else ''

def max_drawdown(prices):
    arr = np.array(prices, dtype=float)
    if len(arr) < 2:
        return ''
    max_dd = 0
    for i in range(len(arr)):
        peak = arr[i]
        if i == len(arr) - 1:
            continue
        trough = np.min(arr[i:])
        dd = (trough - peak) / peak
        if dd < max_dd:
            max_dd = dd
    return max_dd

def calc_period_drawdown(prices, start_date, end_date):
    period_prices = [v for d, v in prices if parse_date(d) >= start_date and parse_date(d) <= end_date]
    return max_drawdown(period_prices) if period_prices else ''

def get_returns(prices):
    arr = np.array([v for _, v in prices])
    return np.diff(arr) / arr[:-1] if len(arr) > 1 else np.array([])

def get_inception(prices):
    if prices:
        return parse_date(prices[0][0])
    return date.max

def get_last(prices):
    if prices:
        return parse_date(prices[-1][0])
    return date.min


def render_html_table(df, output_file='etf_comparison.html'):
    # Color map for ROI columns
    def roi_color(val, vmin, vmax, direction='normal'):
        # Accept float or percent string
        if val == '' or val is None:
            return ''
        if isinstance(val, str) and val.endswith('%'):
            try:
                val = float(val.strip('%')) / 100
            except Exception:
                return ''
        if not isinstance(val, float):
            return ''
        # direction: 'normal' (higher is better), 'inverse' (lower is better)
        if vmax == vmin:
            norm = 0.5
        else:
            if direction == 'inverse':
                norm = (vmax - val) / (vmax - vmin)
            else:
                norm = (val - vmin) / (vmax - vmin)
        # Modern pastel: red (#ff4d4d) to white to green (#4dff88)
        if norm <= 0.5:
            # interpolate red to white
            r = int(0xff * (1 - 2*norm) + 0xff * 2*norm)
            g = int(0x4d * (1 - 2*norm) + 0xff * 2*norm)
            b = int(0x4d * (1 - 2*norm) + 0xff * 2*norm)
        else:
            # interpolate white to green
            r = int(0xff * (2 - 2*norm))
            g = int(0xff * (2 - 2*norm) + 0xff * (2*norm - 1) * (0x88/0xff))
            b = int(0xff * (2 - 2*norm) + 0x88 * (2*norm - 1))
        r = min(max(r, 0), 255)
        g = min(max(g, 0), 255)
        b = min(max(b, 0), 255)
        return f'background-color: rgb({r},{g},{b});'

    # Apply heatmap color-coding to all columns
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    import re
    for col in df.columns:
        vals = []
        for x in df[col]:
            if isinstance(x, float):
                vals.append(x)
            elif isinstance(x, str):
                # Try percent
                if x.endswith('%'):
                    try:
                        vals.append(float(x.strip('%')) / 100)
                    except Exception:
                        vals.append(None)
                else:
                    # Try to extract a number (for Sharpe, Sortino, StdDev)
                    m = re.search(r'-?\d+(?:\.\d+)?', x)
                    if m:
                        try:
                            vals.append(float(m.group(0)))
                        except Exception:
                            vals.append(None)
                    else:
                        vals.append(None)
            else:
                vals.append(None)
        valid_vals = [v for v in vals if v is not None]
        if not valid_vals:
            continue
        vmin, vmax = min(valid_vals), max(valid_vals)
        direction = 'inverse' if (col.lower().startswith('stddev') or 'stddev' in col.lower()) else 'normal'
        for i, v in enumerate(vals):
            styles.at[i, col] = roi_color(v, vmin, vmax, direction=direction) if v is not None else ''

    def style_row(row, style_row):
        return '<tr>' + ''.join([
            f'<td style="{style_row[i]}">{cell}</td>' for i, cell in enumerate(row)
        ]) + '</tr>'

    # Build HTML table manually to inject styles
    html_table = f'<table class="sortable">\n<thead>\n<tr>'
    for col in df.columns:
        html_table += f'<th>{col}</th>'
    html_table += '</tr>\n</thead>\n<tbody>\n'
    for i, row in df.iterrows():
        html_table += style_row(row, styles.iloc[i]) + '\n'
    html_table += '</tbody>\n</table>'

    html = f'''
<html>
<head>
<meta charset="utf-8">
<title>ETF Comparison</title>
<style>
    body {{ font-family: Arial, sans-serif; }}
    .table-container {{
        width: 80%;
        margin: 40px auto 40px auto;
    }}
    table.sortable {{ border-collapse: collapse; width: 100%; }}
    th, td {{ padding: 8px; text-align: right; }}
    th {{ background: #222; color: white; cursor: pointer; }}
    tr:nth-child(even) {{ background: #f2f2f2; }}
    tr:hover {{ background: #e0e0e0; }}
</style>
<script>
// Simple table sort
function sortTable(table, col, num) {{
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);
  const dir = table.getAttribute('data-sort-dir') === 'asc' ? -1 : 1;
  rows.sort((a, b) => {{
    let x = a.cells[col].innerText.replace(/[^-\\d.]/g, ''),
        y = b.cells[col].innerText.replace(/[^-\\d.]/g, '');
    x = num ? parseFloat(x) : x;
    y = num ? parseFloat(y) : y;
    if (x < y) return -1 * dir;
    if (x > y) return 1 * dir;
    return 0;
  }});
  rows.forEach(row => tbody.appendChild(row));
  table.setAttribute('data-sort-dir', dir === 1 ? 'asc' : 'desc');
}}
document.addEventListener('DOMContentLoaded', function() {{
  document.querySelectorAll('table.sortable th').forEach((th, i) => {{
    th.onclick = function() {{
      sortTable(th.closest('table'), i, th.innerText.match(/ROI|Sharpe|Sortino|StdDev|Drawdown|YTD|1y|3y|5y|10y|17y/));
    }};
  }});
}});
</script>
</head>
<body>
<div class="table-container">
{html_table}
</div>
</body>
</html>
'''
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Saved pretty, sortable table to {output_file}')

def main():
    parser = argparse.ArgumentParser(prog='ETF Compare')
    parser.add_argument('etfs', nargs='+', help='List of ETF tickers to compare')
    args = parser.parse_args()

    with open('etfs.json') as f:
        data = json.load(f)

    # Date ranges
    today = date.today()
    ytd = date(today.year, 1, 1)
    periods = {
        'YTD': (ytd, today),
        '1y': (today.replace(year=today.year-1), today),
        '3y': (today.replace(year=today.year-3), today),
        '5y': (today.replace(year=today.year-5), today),
        '10y': (today.replace(year=today.year-10), today),
        '17y': (today.replace(year=today.year-17), today),
    }
    drawdown_periods = [
        ('10/08/2007', '04/19/2010'),
        ('07/20/2015', '04/18/2016'),
        ('10/01/2018', '02/25/2019'),
        ('02/10/2020', '06/01/2020'),
        ('11/08/2021', '07/24/2023'),
        ('02/18/2025', '05/12/2025'),
    ]
    drawdown_periods = [(parse_date(s), parse_date(e)) for s, e in drawdown_periods]

    # Find max shared inception
    inceptions = {etf: get_inception(data[etf]) for etf in args.etfs if etf in data}
    max_inception = max(inceptions.values()) if inceptions else None
    last_dates = {etf: get_last(data[etf]) for etf in args.etfs if etf in data}
    min_last = min(last_dates.values()) if last_dates else None

    # Table header
    header = ['ETF'] + list(periods.keys()) + ['MaxSharedROI', 'Sharpe', 'Sortino', 'StdDev']
    for i, (s, e) in enumerate(drawdown_periods):
        header.append(f'Drawdown {s} to {e}')
    rows = []

    for etf in args.etfs:
        if etf not in data:
            continue
        prices = data[etf]
        row = [etf]
        # ROI for each period
        for _, (start, _) in periods.items():
            roi = calculate_roi(prices, start)
            row.append(roi)
        # Max shared ROI
        if max_inception and min_last:
            roi_shared = calculate_roi(prices, max_inception)
            row.append(roi_shared)
        else:
            row.append('')
        # Sharpe, Sortino, StdDev
        returns = get_returns(prices)
        row.append(sharpe_ratio(returns))
        row.append(sortino_ratio(returns))
        row.append(np.std(returns, ddof=1) * np.sqrt(52) if len(returns) > 1 else '')
        # Drawdowns (skip if ETF inception is after the START of the period)
        inception = get_inception(prices)
        for s, e in drawdown_periods:
            if inception > s:
                row.append('')
            else:
                dd = calc_period_drawdown(prices, s, e)
                row.append(dd)
        rows.append(row)

    # Output as table
    df = pd.DataFrame(rows, columns=header)
    print(df.to_string(index=False))

    # Add max shared ROI date range to the column name
    if max_inception and min_last:
        shared_roi_label = f"MaxSharedROI ({max_inception} to {min_last})"
        df = df.rename(columns={"MaxSharedROI": shared_roi_label})

    # Format columns for display
    percent_cols = [col for col in df.columns if 'ROI' in col or 'Drawdown' in col]
    for col in percent_cols:
        df[col] = df[col].apply(lambda x: f"{x:.2%}" if isinstance(x, float) and x != '' else x)
    if 'Sharpe' in df.columns:
        df['Sharpe'] = df['Sharpe'].apply(lambda x: f"{x:.2f} annualized (weekly)" if isinstance(x, float) and x != '' else x)
    if 'Sortino' in df.columns:
        df['Sortino'] = df['Sortino'].apply(lambda x: f"{x:.2f} annualized (weekly)" if isinstance(x, float) and x != '' else x)
    if 'StdDev' in df.columns:
        df['StdDev'] = df['StdDev'].apply(lambda x: f"{x:.2%} annualized (weekly)" if isinstance(x, float) and x != '' else x)

    # Output as pretty, sortable HTML (no caption, add JS for sorting)
    render_html_table(df, output_file='etf_comparison.html')

if __name__ == '__main__':
    main()
