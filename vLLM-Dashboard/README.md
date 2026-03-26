# vLLM 实时监控看板
<img width="1371" height="850" alt="image" src="https://github.com/user-attachments/assets/aa8825c5-be9d-4957-b037-0338958849bd" />

基于 vLLM `/metrics` 接口的实时监控 Dashboard，支持多 GPU 展示、Token 吞吐量趋势、性能指标追踪。

---

## 功能特性

- **8 个 KPI 指标卡片**：运行中请求数、等待队列、Cache 命中率、TTFT、Prompt/Gen Tokens/s、平均排队时间、累计成功请求数
- **4 张 GPU 仪表盘**：KV Cache 使用率 + 历史趋势折线，每卡独立显示
- **3 组性能趋势图**：TTFT / Cache 命中率 / 排队时间 | Token 吞吐量 | 请求数
- **自动 3 秒刷新**：实时数据无延迟
- **深色主题**：专业监控风格

---

## 快速启动

```powershell
# 克隆或复制本脚本
python C:\vllm_dashboard.py
```

然后打开浏览器访问：`http://127.0.0.1:8055`

---

## 配置

脚本顶部配置区：

```python
VLLM_URL       = "http://10.1.1.X:8001/metrics"  # vLLM metrics 地址
REFRESH_INTERVAL = 3000   # 刷新间隔（毫秒），默认 3 秒
HISTORY_LENGTH   = 60     # 趋势图保留数据点数（约 3 分钟历史）
```

---

## GPU 配置（可自定义）

```python
GPU_LIST = [
    ("0", "GPU 0", "RTX 6000 Ada"),
    ("1", "GPU 1", "RTX 6000 Ada"),
    ("2", "GPU 2", "RTX 6000 Ada"),
    ("3", "GPU 3", "RTX 6000 Ada"),
]
```

如需修改为其他型号或数量，编辑 `update_gpu_gauges()` 函数中的 `GPU_LIST`。

---

## 依赖

```
pip install dash plotly pandas requests
```

---

## 指标说明

| 指标 | 来源 Metric | 说明 |
|------|------------|------|
| 运行中请求 | `vllm:num_requests_running` | 当前正在处理的请求数 |
| 等待队列 | `vllm:num_requests_waiting` | 队列中等待的请求数 |
| Cache 命中率 | `prefix_cache_hits_total / prefix_cache_queries_total` | 前缀缓存命中率 |
| TTFT | `request_time_per_output_token_seconds` 均值 | 首 Token 延迟（ms） |
| Prompt Tokens/s | 增量计算 | 每秒处理的 prompt token 数 |
| Gen Tokens/s | 增量计算 | 每秒生成的 token 数 |
| Avg 排队时间 | `request_queue_time_seconds` 均值 | 平均等待入队时间（ms） |
| 成功请求总数 | `request_success_total{finished_reason="stop"}` | 累计成功完成的请求数 |
| KV Cache | `vllm:kv_cache_usage_perc` | GPU KV Cache 利用率 |

---

## 目录结构

```
C:\Users\gang\.qclaw\workspace\
├── vllm_dashboard.py   # 主程序
├── README.md            # 本文件
└── memory/              # 日常记录目录
```

---

## 常见问题

**Q: 浏览器打开空白或报 JS 错误？**
> Dash 浏览器 JS 有缓存，每次代码更新后换一个新端口启动即可绕过。

**Q: GPU 卡只显示一张？**
> vLLM 的 `/metrics` 以 engine 为粒度导出指标，如果只有 `engine="0"` 则只显示一张卡。如需显示多卡，需修改 `GPU_LIST` 手动配置。

**Q: 百分比一闪而过？**
> 检查 gauge 的 `margin.bottom` 是否足够（建议 >= 20px），以及 HTML 容器 height 是否足够容纳 figure 内容。
