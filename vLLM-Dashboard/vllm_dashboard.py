"""
vLLM 实时监控看板 v2
直接用全局变量在 callbacks 间共享数据，不走 dcc.Store 序列化

用法: python vllm_dashboard.py
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import requests
import re
import time
from collections import deque
import threading
import traceback

# ==================== 配置 ====================
VLLM_URL = "http://10.1.1.X:8001/metrics"
REFRESH_INTERVAL = 3000   # 毫秒
HISTORY_LENGTH  = 60      # 趋势图保留数据点数

# ==================== 全局数据 ====================
data_lock = threading.Lock()

gpu_cache_history  = {}   # {gpu_id: deque of float or None}
throughput_hist    = {"prompt": deque(maxlen=HISTORY_LENGTH), "generation": deque(maxlen=HISTORY_LENGTH)}
request_hist       = {"running": deque(maxlen=HISTORY_LENGTH), "waiting": deque(maxlen=HISTORY_LENGTH)}
perf_hist          = {"ttft": deque(maxlen=HISTORY_LENGTH), "cache_hit": deque(maxlen=HISTORY_LENGTH), "queue_time": deque(maxlen=HISTORY_LENGTH)}
timestamps_hist    = deque(maxlen=HISTORY_LENGTH)

latest = {
    "gpu_cache": {},          # {gpu_id: ratio}
    "num_running": 0,
    "num_waiting": 0,
    "prompt_tokens_sec": 0.0,
    "generation_tokens_sec": 0.0,
    "model_name": "",
    # 扩展指标
    "cache_hit_rate": 0.0,    # 缓存命中率 0~1
    "ttft": 0.0,              # avg time to first token (ms)
    "avg_queue_time": 0.0,    # 平均排队时间 (ms)
    "total_success": 0,       # 累计成功请求数
    "avg_gen_tokens_per_req": 0.0,  # 平均每请求生成 token 数
    "error": None,
    "ts": "",
}

prev_prompt_tokens = None
prev_gen_tokens    = None
prev_time          = None

# ==================== Prometheus 解析 ====================

def parse_metrics(text):
    metrics = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = re.match(r'^([\w:]+)(?:\{([^}]*)\})?\s+([\d.e+-]+)', line)
        if not m:
            continue
        name = m.group(1)
        labels = {}
        for k, v in re.findall(r'([\w]+)="([^"]*)"', m.group(2) or ''):
            labels[k] = v
        metrics[name] = {"labels": labels, "value": float(m.group(3))}
    return metrics


def fetch_and_update():
    global prev_prompt_tokens, prev_gen_tokens, prev_time, latest

    try:
        resp = requests.get(VLLM_URL, timeout=5)
        resp.raise_for_status()
        text = resp.text
    except Exception as e:
        with data_lock:
            latest["error"] = f"连接失败: {e}"
        return

    metrics = parse_metrics(text)
    now = time.time()

    with data_lock:
        latest["error"] = None

        # ---- GPU KV Cache Usage ----
        gpu_cache = {}
        for metric_name in [
            "vllm:kv_cache_usage_perc",
            "vllm:gpu_cache_usage_per_query",
            "vllm:gpu_cache_usage",
            "gpu_cache_usage_per_query",
            "vllm:cache_usage",
        ]:
            if metric_name in metrics:
                labels = metrics[metric_name]["labels"]
                gpu_id = labels.get("engine",
                          labels.get("gpu",
                          labels.get("gpu_id",
                          labels.get("device", "0"))))
                gpu_cache[gpu_id] = metrics[metric_name]["value"]
                # engine=0 单卡，强制注册 gpu_id="0"
                if "engine" in labels:
                    gpu_cache["0"] = gpu_cache.get("0", metrics[metric_name]["value"])
                latest["model_name"] = labels.get("model_name", "")
                break
        else:
            for name, data in metrics.items():
                if ("kv_cache" in name.lower() or "gpu_cache" in name.lower()) and "usage" in name.lower():
                    lbls = data["labels"]
                    gid = lbls.get("engine", lbls.get("gpu", lbls.get("gpu_id", lbls.get("device", "0"))))
                    gpu_cache[gid] = data["value"]

        latest["gpu_cache"] = gpu_cache

        # 初始化各 gpu_id 的历史 deque
        all_ids = set(gpu_cache.keys()) | set(gpu_cache_history.keys())
        for gid in all_ids:
            if gid not in gpu_cache_history:
                gpu_cache_history[gid] = deque(maxlen=HISTORY_LENGTH)

        # ---- 请求数 ----
        num_running = 0
        num_waiting = 0
        for name in ["vllm:num_requests_running", "num_requests_running"]:
            if name in metrics:
                num_running = int(metrics[name]["value"])
        for name in ["vllm:num_requests_waiting", "num_requests_waiting"]:
            if name in metrics:
                num_waiting = int(metrics[name]["value"])
        latest["num_running"] = num_running
        latest["num_waiting"]  = num_waiting

        # ---- Token 吞吐量 ----
        prompt_tokens = 0.0
        gen_tokens    = 0.0
        for name in ["vllm:prompt_tokens_total", "prompt_tokens_total"]:
            if name in metrics:
                prompt_tokens = metrics[name]["value"]
        for name in ["vllm:generation_tokens_total", "generation_tokens_total"]:
            if name in metrics:
                gen_tokens = metrics[name]["value"]

        if prev_time is not None and prev_prompt_tokens is not None:
            dt = now - prev_time
            if dt > 0:
                p_rate = max(0, (prompt_tokens - prev_prompt_tokens) / dt)
                g_rate = max(0, (gen_tokens    - prev_gen_tokens)    / dt)
                latest["prompt_tokens_sec"]    = p_rate
                latest["generation_tokens_sec"] = g_rate
                latest["ts"]                   = time.strftime("%H:%M:%S")

                throughput_hist["prompt"].append(p_rate)
                throughput_hist["generation"].append(g_rate)
                request_hist["running"].append(num_running)
                request_hist["waiting"].append(num_waiting)
                timestamps_hist.append(latest["ts"])

                # ---- 性能指标 ----
                # TTFT: avg request_time_per_output_token (sum/count)
                tpot = metrics.get("vllm:request_time_per_output_token_seconds_sum", {})
                tpot_c = metrics.get("vllm:request_time_per_output_token_seconds_count", {})
                if tpot and tpot_c and tpot_c["value"] > 0:
                    ttft_s = tpot["value"] / tpot_c["value"]
                    latest["ttft"] = ttft_s * 1000   # ms
                    perf_hist["ttft"].append(ttft_s * 1000)

                # 缓存命中率
                pq = metrics.get("vllm:prefix_cache_queries_total", {}).get("value", 0)
                ph = metrics.get("vllm:prefix_cache_hits_total", {}).get("value", 0)
                if pq > 0:
                    cache_rate = ph / pq
                    latest["cache_hit_rate"] = cache_rate
                    perf_hist["cache_hit"].append(cache_rate)

                # 平均排队时间
                qt_sum = metrics.get("vllm:request_queue_time_seconds_sum", {})
                qt_cnt = metrics.get("vllm:request_queue_time_seconds_count", {})
                if qt_sum and qt_cnt and qt_cnt["value"] > 0:
                    avg_qt_s = qt_sum["value"] / qt_cnt["value"]
                    latest["avg_queue_time"] = avg_qt_s * 1000   # ms
                    perf_hist["queue_time"].append(avg_qt_s * 1000)

                # 累计完成请求数：用 *_count 指标（bucket 统计 +Inf = 全部请求）
                total_s = 0
                for name, data in metrics.items():
                    if name == "vllm:request_success_total":
                        total_s += int(data["value"])
                # 如果 success_total 为 0，用 generation_tokens_count 作为代理（更可靠）
                gen_cnt = 0
                for name, data in metrics.items():
                    if name == "vllm:request_generation_tokens_count":
                        gen_cnt = int(data["value"])
                latest["total_success"] = total_s if total_s > 0 else gen_cnt

                # Prefill 平均耗时 (ms)
                pf_sum = 0.0
                pf_cnt = 0.0
                for name, data in metrics.items():
                    if name == "vllm:request_prefill_time_seconds_sum":
                        pf_sum = data["value"]
                    if name == "vllm:request_prefill_time_seconds_count":
                        pf_cnt = data["value"]
                latest["avg_gen_tokens_per_req"] = (pf_sum / pf_cnt * 1000) if pf_cnt > 0 else 0.0

                for gid in gpu_cache_history:
                    gpu_cache_history[gid].append(gpu_cache.get(gid))

        prev_prompt_tokens = prompt_tokens
        prev_gen_tokens    = gen_tokens
        prev_time          = now


# ==================== Dash App ====================

app = dash.Dash(__name__)
app.title = "vLLM 监控看板"

DARK_BG      = "#0d1117"
CARD_BG      = "#161b22"
BORDER       = "#30363d"
TEXT_MAIN    = "#e6edf3"
TEXT_MUTED   = "#7d8590"
ACCENT_BLUE  = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_RED   = "#f85149"
ACCENT_ORANGE= "#d29922"
ACCENT_PURPLE= "#a371f7"


def hex_rgba(hex_color, alpha=0.13):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def make_card(children, **kwargs):
    return html.Div(children, style={
        "background": CARD_BG,
        "border": f"1px solid {BORDER}",
        "borderRadius": "12px",
        "padding": "20px",
        **kwargs
    })


def metric_card(title, value, subtitle="", color=ACCENT_BLUE, icon=""):
    return make_card([
        html.Div(title, style={
            "color": TEXT_MUTED, "fontSize": "12px", "fontWeight": "600",
            "letterSpacing": "0.5px", "textTransform": "uppercase", "marginBottom": "8px"
        }),
        html.Div([
            html.Span(icon, style={"fontSize": "22px", "marginRight": "8px"}),
            html.Span(str(value), style={
                "color": color, "fontSize": "34px", "fontWeight": "700",
                "fontFamily": "JetBrains Mono, Consolas, monospace"
            }),
        ]),
        html.Div(subtitle, style={"color": TEXT_MUTED, "fontSize": "11px", "marginTop": "4px"}),
    ])


app.layout = html.Div([
    dcc.Interval(id="fetch-interval", interval=REFRESH_INTERVAL),

    # ===== Header =====
    html.Div([
        html.Div([
            html.Span("🤖", style={"fontSize": "28px", "marginRight": "12px"}),
            html.Div([
                html.Div("vLLM 实时监控看板", style={"fontSize": "20px", "fontWeight": "700", "color": TEXT_MAIN}),
                html.Div(VLLM_URL, style={
                    "fontSize": "11px", "color": TEXT_MUTED,
                    "fontFamily": "JetBrains Mono, Consolas, monospace", "marginTop": "2px"
                }),
                html.Div(id="model-name-display", style={
                    "fontSize": "11px", "color": ACCENT_BLUE,
                    "fontFamily": "JetBrains Mono, Consolas, monospace", "marginTop": "2px"
                }),
            ]),
        ]),
        html.Div([
            html.Div(id="update-time",    style={"color": TEXT_MUTED, "fontSize": "11px", "textAlign": "right"}),
            html.Div(id="conn-status",    style={"color": ACCENT_GREEN, "fontSize": "12px", "fontWeight": "600", "textAlign": "right", "marginTop": "4px"}),
        ]),
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "20px", "padding": "0 4px"}),

    # ===== Error Banner =====
    html.Div(id="error-banner", children="", style={"display": "none"}),

    # ===== Outer 4-column grid: KPI rows + GPU row，共用同一套列宽 =====
    # 关键：所有子元素 gap=8px，格子宽=(总宽-3×8)/4，两两完全对齐
    html.Div([

        # ---- KPI 组1: Running + Waiting ----
        html.Div([
            html.Div(id="kpi-running"),
            html.Div(id="kpi-waiting"),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"}),

        # ---- KPI 组2: Cache + TTFT ----
        html.Div([
            html.Div(id="kpi-cache"),
            html.Div(id="kpi-ttft"),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"}),

        # ---- KPI 组3: Prompt + Gen TPS ----
        html.Div([
            html.Div(id="kpi-prompt"),
            html.Div(id="kpi-gen"),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"}),

        # ---- KPI 组4: Queue + Success ----
        html.Div([
            html.Div(id="kpi-queue"),
            html.Div(id="kpi-success"),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"}),

        # ---- GPU Section Header ----
        html.Div([
            html.Span("🖥 GPU Monitor", style={"color": ACCENT_BLUE, "fontWeight": "700"}),
            html.Span("  ·  RTX 6000 Ada × 4  ·  Qwen3.5-122B-A10B-FP8",
                      style={"color": TEXT_MUTED, "fontSize": "12px"}),
        ], style={
            "color": TEXT_MUTED, "fontSize": "12px", "fontWeight": "600",
            "letterSpacing": "0.5px", "marginTop": "4px", "marginBottom": "8px",
            "gridColumn": "1 / -1",
        }),

        # ---- GPU Gauges Row ----
        html.Div(id="gpu-gauges-container", children=[
            html.Div("⏳ 正在连接并获取 GPU 信息...",
                     style={"color": TEXT_MUTED, "textAlign": "center", "padding": "40px"})
        ], style={
            "display": "grid",
            "gridTemplateColumns": "repeat(4, 1fr)",
            "gap": "8px",
            "gridColumn": "1 / -1",
        }),

    ], style={
        # 4列 grid，行/列 gap 统一=8px，格子宽完全相同
        "display": "grid",
        "gridTemplateColumns": "repeat(4, 1fr)",
        "gap": "8px",
    }),

    # ===== Row 3: Performance Metrics =====
    html.Div([make_card([
        html.Div("⚡ 性能指标趋势  ·  TTFT / Cache 命中率 / 排队时间", style={
            "color": TEXT_MUTED, "fontSize": "12px", "textTransform": "uppercase",
            "letterSpacing": "0.5px", "marginBottom": "8px", "fontWeight": "600"
        }),
        dcc.Graph(id="perf-chart", config={"displayModeBar": False}),
    ])], style={"marginBottom": "12px"}),

    # ===== Row 4: Trend Charts =====
    html.Div([
        html.Div([make_card([
            html.Div("📊 Token 吞吐量趋势", style={
                "color": TEXT_MUTED, "fontSize": "12px", "textTransform": "uppercase",
                "letterSpacing": "0.5px", "marginBottom": "8px", "fontWeight": "600"
            }),
            dcc.Graph(id="throughput-chart", config={"displayModeBar": False}),
        ])], style={"flex": "1"}),
        html.Div([make_card([
            html.Div("📋 请求数趋势", style={
                "color": TEXT_MUTED, "fontSize": "12px", "textTransform": "uppercase",
                "letterSpacing": "0.5px", "marginBottom": "8px", "fontWeight": "600"
            }),
            dcc.Graph(id="requests-chart", config={"displayModeBar": False}),
        ])], style={"flex": "1"}),
    ], style={"display": "flex", "gap": "12px", "marginBottom": "0"}),

], style={
    "background": DARK_BG, "minHeight": "100vh", "padding": "24px",
    "fontFamily": "Inter, -apple-system, sans-serif", "color": TEXT_MAIN,
    "maxWidth": "1600px", "margin": "0 auto",
})


# ==================== Callbacks ====================

@app.callback(
    Output("conn-status", "children"),
    Output("conn-status", "style"),
    Output("update-time", "children"),
    Output("error-banner", "children"),
    Output("error-banner", "style"),
    Input("fetch-interval", "n_intervals"),
)
def update_status(n):
    fetch_and_update()   # ← 每次定时触发，先抓最新数据
    with data_lock:
        err  = latest["error"]
        ts   = latest["ts"] or time.strftime("%H:%M:%S")

    if err:
        banner = html.Div([
            html.Span("🔴 ", style={"marginRight": "6px"}),
            html.Span(str(err)),
        ], style={"display": "flex", "alignItems": "center"})
        banner_style = {
            "display": "block",
            "background": "#3a1a1a",
            "border": f"1px solid {ACCENT_RED}",
            "borderRadius": "8px", "padding": "10px 16px",
            "color": ACCENT_RED, "fontSize": "13px", "marginBottom": "16px",
        }
        return "🔴 连接异常", {"color": ACCENT_RED, "fontSize": "12px", "fontWeight": "600", "textAlign": "right", "marginTop": "4px"}, f"最后更新: {ts}", banner, banner_style

    banner_style = {"display": "none"}
    return "🟢 运行中", {"color": ACCENT_GREEN, "fontSize": "12px", "fontWeight": "600", "textAlign": "right", "marginTop": "4px"}, f"最后更新: {ts}", "", banner_style


@app.callback(
    Output("model-name-display", "children"),
    Input("fetch-interval", "n_intervals"),
)
def update_model_name(n):
    with data_lock:
        return latest["model_name"]


def make_kpi_card(label, value, subtitle="", color=ACCENT_BLUE, icon=""):
    """单张 KPI 卡片，固定高度撑满格子"""
    return html.Div(make_card([
        html.Div(label, style={
            "color": TEXT_MUTED, "fontSize": "9px", "fontWeight": "600",
            "letterSpacing": "0.3px", "textTransform": "uppercase", "marginBottom": "4px"
        }),
        html.Div([
            html.Span(icon, style={"fontSize": "12px", "marginRight": "4px", "verticalAlign": "middle"}),
            html.Span(str(value), style={
                "color": color, "fontSize": "19px", "fontWeight": "700",
                "fontFamily": "JetBrains Mono, Consolas, monospace", "lineHeight": "1",
            }),
        ], style={"display": "flex", "alignItems": "baseline", "marginBottom": "4px"}),
        html.Div(subtitle, style={
            "color": TEXT_MUTED, "fontSize": "8px",
            "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap",
        }),
    ], style={
        "padding": "10px 8px",
        "height": "82px", "boxSizing": "border-box",
        "display": "flex", "flexDirection": "column", "justifyContent": "center",
    }))


# ── 8 个独立 KPI 回调 ──────────────────────────────────────────
@app.callback(Output("kpi-running","children"), Input("fetch-interval","n_intervals"))
def kpi_running(n):
    with data_lock: v = latest["num_running"]
    c = ACCENT_GREEN if v<10 else ACCENT_ORANGE if v<30 else ACCENT_RED
    return make_kpi_card("运行中请求", f"{v}", "num_requests_running", c, "🔄")

@app.callback(Output("kpi-waiting","children"), Input("fetch-interval","n_intervals"))
def kpi_waiting(n):
    with data_lock: v = latest["num_waiting"]
    c = ACCENT_GREEN if v==0 else ACCENT_ORANGE if v<20 else ACCENT_RED
    return make_kpi_card("等待队列", f"{v}", "num_requests_waiting", c, "⏳")

@app.callback(Output("kpi-cache","children"), Input("fetch-interval","n_intervals"))
def kpi_cache(n):
    with data_lock: v = latest["cache_hit_rate"]
    c = ACCENT_GREEN if v>0.7 else ACCENT_ORANGE if v>0.4 else ACCENT_RED
    return make_kpi_card("Cache 命中率", f"{v*100:.1f}%", "prefix_cache", c, "💾")

@app.callback(Output("kpi-ttft","children"), Input("fetch-interval","n_intervals"))
def kpi_ttft(n):
    with data_lock: v = latest["ttft"]
    return make_kpi_card("TTFT (ms)", f"{v:.1f}", "time to first token", ACCENT_BLUE, "⚡")

@app.callback(Output("kpi-prompt","children"), Input("fetch-interval","n_intervals"))
def kpi_prompt(n):
    with data_lock: v = latest["prompt_tokens_sec"]
    return make_kpi_card("Prompt Tokens/s", f"{v:.0f}", "实时吞吐量", ACCENT_BLUE, "📥")

@app.callback(Output("kpi-gen","children"), Input("fetch-interval","n_intervals"))
def kpi_gen(n):
    with data_lock: v = latest["generation_tokens_sec"]
    return make_kpi_card("Gen Tokens/s", f"{v:.0f}", "实时吞吐量", ACCENT_PURPLE, "📤")

@app.callback(Output("kpi-queue","children"), Input("fetch-interval","n_intervals"))
def kpi_queue(n):
    with data_lock: v = latest["avg_queue_time"]
    c = ACCENT_GREEN if v<100 else ACCENT_ORANGE if v<500 else ACCENT_RED
    return make_kpi_card("Avg 排队 (ms)", f"{v:.1f}", "request_queue_time", c, "🕐")

@app.callback(Output("kpi-success","children"), Input("fetch-interval","n_intervals"))
def kpi_success(n):
    with data_lock: v = latest["total_success"]
    return make_kpi_card("成功请求总数", f"{v:,}", "request_success_total", ACCENT_GREEN, "✅")


@app.callback(
    Output("gpu-gauges-container", "children"),
    Input("fetch-interval", "n_intervals"),
)
def update_gpu_gauges(n):
    # 固定 4 张 RTX 6000 Ada
    GPU_LIST = [
        ("0", "GPU 0", "RTX 6000 Ada"),
        ("1", "GPU 1", "RTX 6000 Ada"),
        ("2", "GPU 2", "RTX 6000 Ada"),
        ("3", "GPU 3", "RTX 6000 Ada"),
    ]

    with data_lock:
        gpu_cache = dict(latest["gpu_cache"])
        cache_val = gpu_cache.get("0", 0.0) if gpu_cache else 0.0
        shared_hist = list(gpu_cache_history.get("0", []))
        valid_count = len([x for x in shared_hist if x is not None])

    val = cache_val
    g_color = ACCENT_GREEN if val < 0.5 else (ACCENT_ORANGE if val < 0.8 else ACCENT_RED)
    line_color = ACCENT_BLUE if val < 0.7 else (ACCENT_ORANGE if val < 0.9 else ACCENT_RED)

    # 共享：共享历史折线
    hist_y = [y for y in shared_hist if y is not None]
    hist_x = list(range(len(hist_y)))
    hist_fig_base = go.Figure()
    hist_fig_base.add_trace(go.Scatter(
        x=hist_x, y=hist_y,
        mode="lines", line=dict(color=line_color, width=1.5),
        fill="tozeroy", fillcolor=hex_rgba(line_color),
    ))
    hist_fig_base.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        margin=dict(l=0, r=0, t=4, b=0),
        height=48,
        xaxis=dict(showgrid=False, showline=False, ticks="",
                   showticklabels=False, zeroline=False, fixedrange=True),
        yaxis=dict(showgrid=True, gridcolor=hex_rgba(BORDER, 0.5),
                   showline=False, zeroline=False,
                   range=[0, 1.05], ticks="", showticklabels=False, fixedrange=True),
    )

    # 共享：仪表盘模板（共用一个基础布局）
    def make_gauge_fig(gid_label):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val * 100,
            number={
                "suffix": "%",
                "font": {"size": 26, "color": g_color,
                         "family": "JetBrains Mono, Consolas, monospace"},
                "valueformat": ".1f",
            },
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 100], "ticks": "", "showticklabels": False},
                "bar": {"color": g_color, "thickness": 0.22},
                "steps": [
                    {"range": [0, 50],   "color": "#1a3a1a"},
                    {"range": [50, 80],  "color": "#3a2a00"},
                    {"range": [80, 100], "color": "#3a1a1a"},
                ],
                "borderwidth": 0,
                "bgcolor": "rgba(0,0,0,0)",
            },
            domain={"x": [0, 1], "y": [0, 1]},
        ))
        fig.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            margin=dict(l=0, r=0, t=8, b=20),
            height=100,
        )
        return fig

    # 构建 4 个 GPU 卡片
    CARD_STYLE = {
        "background": CARD_BG,
        "border": f"1px solid {BORDER}",
        "borderRadius": "12px",
        "padding": "6px 10px",
        "display": "flex",
        "flexDirection": "column",
        "gap": "2px",
        "width": "100%",
        "height": "212px",
        "boxSizing": "border-box",
    }

    children = []
    for gid, gid_label, model_name in GPU_LIST:
        gauge_f  = make_gauge_fig(gid_label)
        hist_f   = go.Figure(hist_fig_base)   # 克隆

        children.append(html.Div([
            # 标题行 (30px)
            html.Div([
                html.Div(gid_label, style={
                    "fontSize": "13px", "fontWeight": "800",
                    "color": ACCENT_BLUE,
                    "fontFamily": "JetBrains Mono, Consolas, monospace",
                }),
                html.Div(model_name, style={
                    "fontSize": "10px", "color": TEXT_MUTED, "marginTop": "1px",
                }),
            ], style={"height": "30px", "marginBottom": "2px", "display": "flex", "flexDirection": "column", "justifyContent": "center"}),

            # 仪表盘 (90px)
            html.Div(dcc.Graph(
                figure=gauge_f,
                config={"displayModeBar": False},
                style={"height": "90px"},
            ), style={"height": "90px", "padding": "0"}),

            # 趋势图 (48px)
            html.Div(dcc.Graph(
                figure=hist_f,
                config={"displayModeBar": False},
                style={"height": "48px"},
            ), style={"height": "48px", "padding": "0"}),

            # 底部状态行 (22px)
            html.Div([
                html.Span("KV Cache", style={
                    "fontSize": "10px", "color": TEXT_MUTED,
                }),
                html.Span(f" · {valid_count} 样本",
                          style={"fontSize": "10px", "color": TEXT_MUTED}),
            ], style={
                "display": "flex", "justifyContent": "space-between",
                "height": "20px", "alignItems": "center",
            }),
        ], style=CARD_STYLE))

    return children


@app.callback(
    Output("perf-chart",       "figure"),
    Output("throughput-chart", "figure"),
    Output("requests-chart",    "figure"),
    Input("fetch-interval", "n_intervals"),
)
def update_charts(n):
    with data_lock:
        ts_list   = list(timestamps_hist)
        prompt_h  = list(throughput_hist["prompt"])
        gen_h     = list(throughput_hist["generation"])
        running_h = list(request_hist["running"])
        waiting_h = list(request_hist["waiting"])
        ttft_h    = list(perf_hist["ttft"])
        cache_h   = list(perf_hist["cache_hit"])
        queue_h   = list(perf_hist["queue_time"])

    empty_fig = go.Figure()
    empty_fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        xaxis=dict(showgrid=True, gridcolor=BORDER, color=TEXT_MUTED, ticks="", showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor=BORDER, color=TEXT_MUTED),
        font={"color": TEXT_MUTED}, height=200,
        margin=dict(l=50, r=20, t=20, b=40),
    )

    # ---- 性能图表 ----
    fig_perf = go.Figure()
    if ttft_h:
        fig_perf.add_trace(go.Scatter(
            x=list(range(len(ttft_h))), y=ttft_h,
            mode="lines", name="TTFT (ms)",
            line=dict(color=ACCENT_BLUE, width=2),
            fill="tozeroy", fillcolor=hex_rgba(ACCENT_BLUE),
        ))
    if cache_h:
        fig_perf.add_trace(go.Scatter(
            x=list(range(len(cache_h))), y=[x*100 for x in cache_h],  # 百分比
            mode="lines", name="Cache 命中率 %",
            line=dict(color=ACCENT_GREEN, width=2),
            fill="tozeroy", fillcolor=hex_rgba(ACCENT_GREEN),
        ))
    if queue_h:
        fig_perf.add_trace(go.Scatter(
            x=list(range(len(queue_h))), y=queue_h,
            mode="lines", name="排队时间 (ms)",
            line=dict(color=ACCENT_ORANGE, width=2),
            fill="tozeroy", fillcolor=hex_rgba(ACCENT_ORANGE),
        ))
    fig_perf.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font={"color": TEXT_MAIN, "family": "Inter, sans-serif"},
        margin=dict(l=50, r=20, t=20, b=40), height=160,
        xaxis={"showgrid": True, "gridcolor": BORDER, "showline": False,
               "color": TEXT_MUTED, "tickfont": {"size": 10}, "ticks": "", "showticklabels": False},
        yaxis={"title": "ms / %", "showgrid": True, "gridcolor": hex_rgba(BORDER, 0.33),
               "showline": False, "color": TEXT_MUTED, "tickfont": {"size": 10}},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font={"color": TEXT_MUTED, "size": 11}),
        hovermode="x unified",
    )

    if not ts_list:
        return empty_fig, empty_fig, empty_fig

    # throughput
    fig_tp = go.Figure()
    fig_tp.add_trace(go.Scatter(
        x=list(range(len(prompt_h))), y=prompt_h,
        mode="lines", name="Prompt Tokens/s",
        line=dict(color=ACCENT_BLUE, width=2),
        fill="tozeroy", fillcolor=hex_rgba(ACCENT_BLUE),
    ))
    fig_tp.add_trace(go.Scatter(
        x=list(range(len(gen_h))), y=gen_h,
        mode="lines", name="Gen Tokens/s",
        line=dict(color=ACCENT_PURPLE, width=2),
        fill="tozeroy", fillcolor=hex_rgba(ACCENT_PURPLE),
    ))
    fig_tp.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font={"color": TEXT_MAIN, "family": "Inter, sans-serif"},
        margin=dict(l=50, r=20, t=20, b=40), height=200,
        xaxis={"showgrid": True, "gridcolor": BORDER, "showline": False,
               "color": TEXT_MUTED, "tickfont": {"size": 10}, "ticks": "", "showticklabels": False},
        yaxis={"title": "tokens/sec", "showgrid": True, "gridcolor": hex_rgba(BORDER, 0.33),
               "showline": False, "color": TEXT_MUTED, "tickfont": {"size": 10}},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font={"color": TEXT_MUTED, "size": 11}),
        hovermode="x unified",
    )

    # requests
    fig_req = go.Figure()
    fig_req.add_trace(go.Scatter(
        x=list(range(len(running_h))), y=running_h,
        mode="lines", name="Running",
        line=dict(color=ACCENT_GREEN, width=2),
        fill="tozeroy", fillcolor=hex_rgba(ACCENT_GREEN),
    ))
    fig_req.add_trace(go.Scatter(
        x=list(range(len(waiting_h))), y=waiting_h,
        mode="lines", name="Waiting",
        line=dict(color=ACCENT_ORANGE, width=2),
        fill="tozeroy", fillcolor=hex_rgba(ACCENT_ORANGE),
    ))
    fig_req.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font={"color": TEXT_MAIN, "family": "Inter, sans-serif"},
        margin=dict(l=50, r=20, t=20, b=40), height=200,
        xaxis={"showgrid": True, "gridcolor": BORDER, "showline": False,
               "color": TEXT_MUTED, "tickfont": {"size": 10}, "ticks": "", "showticklabels": False},
        yaxis={"title": "请求数", "showgrid": True, "gridcolor": hex_rgba(BORDER, 0.33),
               "showline": False, "color": TEXT_MUTED, "tickfont": {"size": 10},
               "rangemode": "tozero"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font={"color": TEXT_MUTED, "size": 11}),
        hovermode="x unified",
    )

    return fig_perf, fig_tp, fig_req


if __name__ == "__main__":
    print(f"启动 vLLM 监控看板 ...")
    print(f"数据源: {VLLM_URL}")
    print(f"打开浏览器访问: http://127.0.0.1:8055")
    app.run(host="0.0.0.0", port=8055, debug=False)
