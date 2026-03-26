# LLM-Ops | 大模型运维工具集

这个仓库主要存放我在本地部署和优化大语言模型（LLM）过程中开发的自动化脚本、监控面板及运维工具。侧重于 GPU 资源管理、vLLM 性能监控以及 AI Agent 框架的生产级落地。

## 🛠️ 项目列表

### 1. [vLLM-Dashboard](./vLLM-Dashboard)
**功能：** 基于 Docker 的 vLLM 实时监控面板。
- **核心指标：** 实时监控显存占用、Token 吞吐量（Generation/Prompt）、请求队列状态。
- **适用场景：** 监控本地多卡环境（如 RTX 6000 Ada / 5090）下的 vLLM 服务稳定性。
- **快速启动：** 见子目录下的 [快速开始指南](./vLLM-Dashboard/README.md)。

---

## 🏗️ 硬件环境参考
本仓库中的工具主要针对以下环境进行优化：
- **GPU:** NVIDIA RTX 6000 Ada / RTX 5090 / H3C & Lenovo 企业级服务器
- **OS:** Ubuntu / Debian (支持 WSL2 节点扩展)
- **Runtime:** Docker / NVIDIA Container Toolkit
- **Inference Engine:** vLLM, SGLang

## 📅 更新计划
- [ ] 增加 OpenClaw 自动心跳检测脚本
- [ ] 多节点 GPU 资源聚合监控看板
- [ ] 自动化模型转换与量化工具 (AWQ/FP8)

## 🤝 交流与贡献
如果你在使用这些工具有任何问题，欢迎提交 Issue。
