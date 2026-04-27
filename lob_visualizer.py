"""
LOB Visualizer — AAPL & CSCO (LOBSTER format)
Run with: py lob_visualizer.py
Then open http://127.0.0.1:8050 in your browser.

Controls
--------
• Date         : select trading day
• LOB Depth    : 1–10 levels shown in the depth chart
• Time slider  : scrub through 9:30 – 16:00
"""

import os
import glob
import re
import bisect

import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

TICKER_DIRS = {
    "AAPL": os.path.join(BASE, "Data", "AAPL_2023-07-01_2023-07-31_10",
                         "output-2023-07", "0", "0", "13"),
    "CSCO": os.path.join(BASE, "Data", "CSCO_2023-07-01_2023-07-31_10",
                         "output-2023-07", "0", "0", "75"),
}

from lobster_data import OB_COLS, OB_PRICE_COLS, load_lobster

_OPEN, _CLOSE = 34_200.0, 57_600.0   # seconds from midnight

TIME_MARKS = {
    34200: "9:30",
    36000: "10:00",
    39600: "11:00",
    43200: "12:00",
    46800: "13:00",
    50400: "14:00",
    54000: "15:00",
    57600: "16:00",
}


# ── Data discovery ────────────────────────────────────────────────────────────

def _find_files(ticker: str) -> dict[str, dict[str, str]]:
    """Return {date: {'ob': path, 'msg': path or None}} for a ticker."""
    d = TICKER_DIRS[ticker]
    ob_files = glob.glob(os.path.join(d, f"{ticker}_*_orderbook_10.csv"))
    result: dict[str, dict[str, str]] = {}
    for ob in ob_files:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(ob))
        if not m:
            continue
        date = m.group(1)
        msg_path = ob.replace("_orderbook_10.csv", "_message_10.csv")
        result[date] = {
            "ob":  ob,
            "msg": msg_path if os.path.exists(msg_path) else None,
        }
    return result


FILES: dict[str, dict[str, dict[str, str]]] = {
    t: _find_files(t) for t in ("AAPL", "CSCO")
}

# All dates where at least one ticker has an orderbook file
all_dates = sorted(set(FILES["AAPL"]) | set(FILES["CSCO"]))


def _date_label(date: str) -> str:
    has_msg = {t: FILES[t].get(date, {}).get("msg") is not None
               for t in ("AAPL", "CSCO")}
    suffix = ""
    if not has_msg["AAPL"] and has_msg["CSCO"]:
        suffix = " (AAPL: OB only)"
    elif has_msg["AAPL"] and not has_msg["CSCO"]:
        suffix = " (CSCO: OB only)"
    elif not has_msg["AAPL"] and not has_msg["CSCO"]:
        suffix = " (OB only)"
    return date + suffix


# ── Data loading & caching ────────────────────────────────────────────────────

_cache: dict = {}


def load_day(ticker: str, date: str):
    """
    Load data for one ticker/date (results are memory-cached).

    Returns
    -------
    times        : np.ndarray  float seconds from midnight (for the time slider)
    ob           : pd.DataFrame  orderbook (columns: ask_p{i}, ask_v{i}, …)
    has_real_time: bool  True when a message file was available
    msg          : pd.DataFrame | None  full message frame from load_lobster
                   (has datetime 'timestamp' and USD 'price'); None for OB-only days
    """
    key = (ticker, date)
    if key in _cache:
        return _cache[key]

    info = FILES.get(ticker, {}).get(date)
    if info is None:
        raise FileNotFoundError(f"No data for {ticker} on {date}")

    has_real_time = info["msg"] is not None

    if has_real_time:
        msg, ob = load_lobster(info["msg"], info["ob"], date)
        # Float seconds from midnight — used by the time slider and bisect search
        t0    = pd.Timestamp(date)
        times = (msg["timestamp"] - t0).dt.total_seconds().values
        # Attach float-second column so chart builders can use it directly
        msg   = msg.assign(time_sec=times)
        n     = min(len(times), len(ob))
        times, ob, msg = times[:n], ob.iloc[:n].reset_index(drop=True), \
                         msg.iloc[:n].reset_index(drop=True)
    else:
        # OB-only day: read book, synthesise an approximate time axis
        ob    = pd.read_csv(info["ob"], header=None, names=OB_COLS)
        ob[OB_PRICE_COLS] = ob[OB_PRICE_COLS] / 10_000
        n     = len(ob)
        times = np.linspace(_OPEN, _CLOSE, n)
        msg   = None

    _cache[key] = (times, ob, has_real_time, msg)
    return times, ob, has_real_time, msg


def lob_snapshot(times: np.ndarray, ob: pd.DataFrame, t: float) -> pd.Series:
    idx = bisect.bisect_right(times, t) - 1
    idx = max(0, min(idx, len(ob) - 1))
    return ob.iloc[idx]


def mid_price_series(times: np.ndarray, ob: pd.DataFrame,
                     n_pts: int = 2_000) -> tuple[np.ndarray, np.ndarray]:
    mid = (ob["ask_p1"].values + ob["bid_p1"].values) / 2.0
    if len(times) > n_pts:
        idx = np.linspace(0, len(times) - 1, n_pts, dtype=int)
        return times[idx], mid[idx]
    return times, mid


def seconds_to_hms(s: float) -> str:
    s = float(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{sec:04.1f}"


# ── Chart builders ────────────────────────────────────────────────────────────

_COLORS = {
    "ask_fill": "rgba(239,68,68,0.18)",
    "ask_line": "#ef4444",
    "bid_fill": "rgba(59,130,246,0.18)",
    "bid_line": "#3b82f6",
    "bg":       "#0f172a",
    "panel":    "#1e293b",
    "grid":     "#334155",
    "text":     "#e2e8f0",
    "subtext":  "#94a3b8",
    "cursor":   "#fbbf24",
    "mid_line": "#94a3b8",
    "price_ln": "#38bdf8",
    "buy_dot":  "#4ade80",
    "sell_dot": "#f87171",
}

_LAYOUT_BASE = dict(
    paper_bgcolor=_COLORS["bg"],
    plot_bgcolor=_COLORS["panel"],
    font=dict(color=_COLORS["text"]),
    margin=dict(l=55, r=15, t=42, b=40),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font=dict(size=11)),
)


def build_lob_chart(row: pd.Series, depth: int, ticker: str,
                    has_real_time: bool) -> go.Figure:
    """Cumulative depth chart (step-function), bids blue / asks red."""
    ask_px = [row[f"ask_p{i}"] for i in range(1, depth + 1)]
    ask_sz = [row[f"ask_v{i}"] for i in range(1, depth + 1)]
    bid_px = [row[f"bid_p{i}"] for i in range(1, depth + 1)]
    bid_sz = [row[f"bid_v{i}"] for i in range(1, depth + 1)]

    ask_cum = list(np.cumsum(ask_sz))
    bid_cum = list(np.cumsum(bid_sz))
    mid     = (ask_px[0] + bid_px[0]) / 2.0

    title_txt = ticker
    if not has_real_time:
        title_txt += "  <sup style='color:#f97316;font-size:11px'>OB only · approx time</sup>"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ask_px, y=ask_cum, mode="lines", name="Ask",
        line=dict(color=_COLORS["ask_line"], width=2, shape="hv"),
        fill="tozeroy", fillcolor=_COLORS["ask_fill"],
        hovertemplate="Ask %{x:.2f}  qty=%{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=bid_px, y=bid_cum, mode="lines", name="Bid",
        line=dict(color=_COLORS["bid_line"], width=2, shape="hv"),
        fill="tozeroy", fillcolor=_COLORS["bid_fill"],
        hovertemplate="Bid %{x:.2f}  qty=%{y:,}<extra></extra>",
    ))
    fig.add_vline(x=mid, line=dict(color=_COLORS["mid_line"], width=1, dash="dot"))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title_txt, font=dict(size=14), x=0.5),
        xaxis=dict(title="Price ($)", gridcolor=_COLORS["grid"],
                   tickformat=".2f", tickfont=dict(size=10)),
        yaxis=dict(title="Cumulative Volume",
                   gridcolor=_COLORS["grid"], tickfont=dict(size=10)),
    )
    return fig


def build_price_chart(times: np.ndarray, mids: np.ndarray,
                      msg_df, current_t: float,
                      ticker: str, has_real_time: bool) -> go.Figure:
    """Mid-price line + trade dots + time cursor."""
    title_txt = ticker
    if not has_real_time:
        title_txt += "  <sup style='color:#f97316;font-size:11px'>approx time</sup>"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=mids, mode="lines", name="Mid",
        line=dict(color=_COLORS["price_ln"], width=1.5),
        hovertemplate="%{y:.4f}<extra></extra>",
    ))

    # Trade executions (event_type 4 or 5) — only available with real message data.
    # msg_df comes from load_lobster via load_day:
    #   'time_sec' = float seconds from midnight  (x-axis)
    #   'price'    = USD, already normalised       (y-axis)
    if msg_df is not None and len(msg_df) > 0:
        trades = msg_df[msg_df["event_type"].isin([4, 5])]
        if len(trades) > 0:
            if len(trades) > 5_000:
                trades = trades.iloc[np.linspace(0, len(trades) - 1,
                                                 5_000, dtype=int)]
            fig.add_trace(go.Scatter(
                x=trades["time_sec"].values,
                y=trades["price"].values,       # already in USD
                mode="markers", name="Trade",
                marker=dict(
                    symbol="circle", size=4,
                    color=np.where(trades["direction"].values == 1,
                                   _COLORS["buy_dot"], _COLORS["sell_dot"]),
                    line=dict(width=0),
                ),
                hovertemplate="Trade %{y:.4f}<extra></extra>",
            ))

    fig.add_vline(x=current_t,
                  line=dict(color=_COLORS["cursor"], width=1.5, dash="dash"))

    x_ticks = dict(
        tickvals=list(TIME_MARKS.keys()),
        ticktext=list(TIME_MARKS.values()),
    ) if has_real_time else {}

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text=title_txt, font=dict(size=14), x=0.5),
        xaxis=dict(title="Time", gridcolor=_COLORS["grid"],
                   tickfont=dict(size=10), **x_ticks),
        yaxis=dict(title="Price ($)", gridcolor=_COLORS["grid"],
                   tickformat=".2f", tickfont=dict(size=10)),
    )
    return fig


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color=_COLORS["subtext"]))
    fig.update_layout(**_LAYOUT_BASE,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# ── App layout ────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="LOB Visualizer")

app.layout = html.Div(
    style={"backgroundColor": _COLORS["bg"], "minHeight": "100vh",
           "fontFamily": "'Inter','Segoe UI',sans-serif",
           "color": _COLORS["text"], "padding": "16px 20px"},
    children=[

        # Header
        html.H1("Limit Order Book Visualizer",
                style={"textAlign": "center", "margin": "0 0 3px",
                       "fontSize": "1.55rem", "fontWeight": 700,
                       "color": "#f8fafc"}),
        html.P("AAPL & CSCO · July 2023 · LOBSTER data",
               style={"textAlign": "center", "color": _COLORS["subtext"],
                      "margin": "0 0 18px", "fontSize": "0.82rem"}),

        # Controls
        html.Div(
            style={"display": "flex", "gap": "28px", "alignItems": "flex-start",
                   "flexWrap": "wrap", "backgroundColor": _COLORS["panel"],
                   "borderRadius": "10px", "padding": "14px 22px 16px",
                   "marginBottom": "18px", "boxShadow": "0 2px 12px #0006"},
            children=[

                # ── Date dropdown ──────────────────────────────────────────
                html.Div([
                    html.Label("Trading Date",
                               style={"fontSize": "0.78rem",
                                      "color": _COLORS["subtext"],
                                      "marginBottom": "6px",
                                      "display": "block",
                                      "fontWeight": 600}),
                    dcc.Dropdown(
                        id="date-picker",
                        options=[{"label": _date_label(d), "value": d}
                                 for d in all_dates],
                        value=all_dates[0],
                        clearable=False,
                        style={"width": "210px"},
                    ),
                ]),

                # ── Depth slider ───────────────────────────────────────────
                html.Div([
                    html.Label(id="depth-label",
                               children="LOB Depth: 5 levels",
                               style={"fontSize": "0.78rem",
                                      "color": _COLORS["subtext"],
                                      "marginBottom": "6px",
                                      "display": "block",
                                      "fontWeight": 600}),
                    dcc.Slider(
                        id="depth-slider",
                        min=1, max=10, step=1, value=5,
                        marks={i: str(i) for i in range(1, 11)},
                        tooltip={"always_visible": False},
                        updatemode="drag",
                    ),
                ], style={"flex": "1", "minWidth": "280px"}),

                # ── Time slider ────────────────────────────────────────────
                html.Div([
                    html.Label(id="time-label",
                               children="Time: 09:30:0",
                               style={"fontSize": "0.78rem",
                                      "color": _COLORS["subtext"],
                                      "marginBottom": "6px",
                                      "display": "block",
                                      "fontWeight": 600}),
                    dcc.Slider(
                        id="time-slider",
                        min=_OPEN, max=_CLOSE, step=30,
                        value=_OPEN,
                        marks=TIME_MARKS,
                        tooltip={"always_visible": False},
                        updatemode="drag",
                    ),
                ], style={"flex": "4", "minWidth": "380px"}),
            ],
        ),

        # LOB depth charts
        html.H3("Order Book Depth Snapshot",
                style={"margin": "0 0 8px", "fontSize": "0.95rem",
                       "color": _COLORS["subtext"], "fontWeight": 600,
                       "letterSpacing": "0.03em"}),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                   "gap": "14px", "marginBottom": "18px"},
            children=[
                dcc.Graph(id="ob-aapl", config={"displayModeBar": False},
                          style={"height": "310px"}),
                dcc.Graph(id="ob-csco", config={"displayModeBar": False},
                          style={"height": "310px"}),
            ],
        ),

        # Mid-price charts
        html.H3("Mid-Price History  ·  green dot = buy trade, red dot = sell trade",
                style={"margin": "0 0 8px", "fontSize": "0.95rem",
                       "color": _COLORS["subtext"], "fontWeight": 600,
                       "letterSpacing": "0.03em"}),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                   "gap": "14px"},
            children=[
                dcc.Graph(id="price-aapl", config={"displayModeBar": False},
                          style={"height": "270px"}),
                dcc.Graph(id="price-csco", config={"displayModeBar": False},
                          style={"height": "270px"}),
            ],
        ),

        # Legend note
        html.P(
            "⚑ Dates marked 'OB only' have no message file; time axis is approximate (linear interpolation).",
            style={"color": "#f97316", "fontSize": "0.75rem",
                   "marginTop": "14px", "textAlign": "center"},
        ),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("depth-label", "children"),
    Input("depth-slider",  "value"),
)
def _depth_label(depth):
    return f"LOB Depth: {depth} level{'s' if depth != 1 else ''}"


@app.callback(
    Output("time-label", "children"),
    Input("time-slider",  "value"),
)
def _time_label(t):
    return f"Time: {seconds_to_hms(t)}"


@app.callback(
    Output("ob-aapl",    "figure"),
    Output("ob-csco",    "figure"),
    Output("price-aapl", "figure"),
    Output("price-csco", "figure"),
    Input("date-picker",  "value"),
    Input("depth-slider", "value"),
    Input("time-slider",  "value"),
)
def update_charts(date: str, depth: int, t: float):
    out_ob    = {}
    out_price = {}

    for ticker in ("AAPL", "CSCO"):
        try:
            times, ob, has_real_time, msg_df = load_day(ticker, date)
        except FileNotFoundError:
            out_ob[ticker]    = _empty_fig(f"No data for {ticker} on {date}")
            out_price[ticker] = _empty_fig(f"No data for {ticker} on {date}")
            continue

        row              = lob_snapshot(times, ob, t)
        t_arr, mid_arr   = mid_price_series(times, ob)

        out_ob[ticker]    = build_lob_chart(row, depth, ticker, has_real_time)
        out_price[ticker] = build_price_chart(t_arr, mid_arr, msg_df,
                                              t, ticker, has_real_time)

    return (out_ob.get("AAPL",    _empty_fig("AAPL")),
            out_ob.get("CSCO",    _empty_fig("CSCO")),
            out_price.get("AAPL", _empty_fig("AAPL")),
            out_price.get("CSCO", _empty_fig("CSCO")))


# ── Dark-theme CSS ─────────────────────────────────────────────────────────────
app.index_string = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        body { margin:0; background:#0f172a; }
        /* Sliders */
        .rc-slider-rail               { background:#334155 !important; }
        .rc-slider-track              { background:#3b82f6 !important; }
        .rc-slider-handle             { background:#60a5fa !important;
                                        border-color:#3b82f6 !important;
                                        opacity:1 !important; }
        .rc-slider-handle:hover,
        .rc-slider-handle:focus       { border-color:#93c5fd !important;
                                        box-shadow:0 0 0 5px rgba(59,130,246,.25) !important; }
        .rc-slider-mark-text          { color:#64748b !important; font-size:11px; }
        /* Dropdown */
        .Select-control               { background:#0f172a !important;
                                        border-color:#334155 !important; }
        .Select-value-label,
        .Select-input > input         { color:#e2e8f0 !important; }
        .Select-menu-outer            { background:#1e293b !important;
                                        border-color:#334155 !important; }
        .Select-option                { background:#1e293b !important;
                                        color:#e2e8f0 !important; }
        .Select-option.is-focused     { background:#334155 !important; }
        .Select-arrow                 { border-color:#64748b transparent transparent !important; }
        .Select-placeholder           { color:#64748b !important; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

if __name__ == "__main__":
    print("LOB Visualizer -> http://127.0.0.1:8050")
    app.run(debug=False, host="127.0.0.1", port=8050)
