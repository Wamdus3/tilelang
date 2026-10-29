# TileLang Harness

用于 TileLang 算子生成、独立评测和可重放实验的确定性框架。当前版本为 **0.5.0**，优先支持 Ascend。
控制器负责状态、预算和证据，外部 agent runtime 提交计划和源码；不需要常驻统筹 agent。

## 文档入口

| 文档 | 内容 |
| --- | --- |
| [架构与模块连接](docs/architecture.zh-CN.md) | 控制边界、模块职责、数据流、状态和身份 |
| [实现状态与验证](docs/implementation-status.zh-CN.md) | P0/P1/P2 已实现内容、测试边界和未完成项 |
| [后续计划](docs/roadmap.zh-CN.md) | 生成对比、profiling、Autotune、Macro 和知识演进 |
| [P1/P2 操作指南](P1-P2-experiments.zh-CN.md) | file/command 交接、四阶段 Skill、预算、修复和单 case 设备入口 |
| [SSD 任务接口](SSD-experiment.zh-CN.md) | shape、dtype、ABI、数值合同和参考 |
| [设备验证说明](A5-validation.zh-CN.md) | 环境检查、调用预算和证据层级 |
| [离线环境准备](environments/a5/README.zh-CN.md) | 本地制包、ELF ABI 审计、显式前缀部署 |

文档使用仓库相对链接；原始实验目录、数据库、模型响应、编译产物及个人环境留在工作区，不随源码提交。

## 快速开始

主机 CLI 只依赖 NumPy，不导入 TileLang、PyTorch 或 torch_npu。执行和恢复使用 Linux 进程与文件锁；
Windows 用户在本地 WSL 运行。以下命令从仓库根目录执行，不访问设备：

```bash
python -m pip install -e './harness[test]'
python -m tilelang_harness list
python -m tilelang_harness validate vecadd
python -m tilelang_harness selfcheck vecadd
python -m pytest harness/tests -q
python -m ruff check harness
```

不安装包时，设置 `PYTHONPATH="$PWD/harness/src:$PWD"`，并使用已有 Python 环境。

## 内置任务

| 任务 | 合同 |
| --- | --- |
| vecadd | FP32 加法，包含尾块；7 个固定 case |
| rmsnorm | 输出归一化结果与每行 rstd；6 个固定 case |
| fft_c2c | complex64 pilot，使用独立实部/虚部 FP32 缓冲；8 个 case |
| fft_c2c_complex64 | TileOPs 固定提交中的 8 条 shape/dtype 配对 |
| fft_c2c_complex128 | 同一来源中的 2 条 FP64 配对；主机参考/lowering 合同，不代表设备执行支持 |
| ssd_chunk_state_fwd_* | FP16/BF16 × plain/seq 四项任务，共 6 条上游 workload |

TaskSpec 冻结语义、精度、输入种子、必需输出、case 集合、oracle、基线源码与测量协议。
范围和来源记录在 [fixtures](src/tilelang_harness/fixtures)。新增任务需要独立 oracle 与基线，不能把生成候选当标杆。

## P1：冻结候选、评测与重放

```bash
python -m tilelang_harness seed vecadd --destination output/vecadd-source
python -m tilelang_harness init vecadd --run-dir output/vecadd-run --max-attempts 10
python -m tilelang_harness submit output/vecadd-source --run-dir output/vecadd-run
# 将 submit 返回的 CandidateID 赋给 CID：
python -m tilelang_harness run "$CID" --run-dir output/vecadd-run --mode lower --timeout-s 300
python -m tilelang_harness report --run-dir output/vecadd-run --output output/vecadd-report.json
python -m tilelang_harness resume --run-dir output/vecadd-run
# 将 report 中的 AttemptID 赋给 AID；重放消耗一次新作业预算：
python -m tilelang_harness replay "$AID" --run-dir output/vecadd-run
```

`lower` 需要本地已构建的 Ascend TileLang 环境，保存 input TIR、lowered TIR、ASC 和 BuildID。
构建遵循 [tilelang-build](../.agents/skills/tilelang-build/SKILL.md)，只在本地 WSL 进行。
`seed --abi numpy` 与 `run --mode host` 用于主机诊断；两种模式均不产生设备正确性或性能结论。

CandidateID 绑定整个源码 bundle 和配置；BuildID 绑定每个 case 的实际编译输入与产物；
评测记录再绑定任务、协议和本次作业。SQLite 为权威元数据，CAS 保存不可变证据，JSON 为导出视图。
独立 watchdog 覆盖导入、编译与执行的墙钟期限；恢复先核对旧作业，不隐式重试设备调用。

## P2：通过原 Skill 生成

```bash
python -m tilelang_harness experiment-init harness/examples/generation-comparison.json \
  --repo-root "$PWD" --run-dir output/generation-study
python -m tilelang_harness experiment-run --run-dir output/generation-study
```

模板的模型名称是占位符，需要按实际 runtime 填写。file 方式返回请求目录，外部 agent 按本次
`context.json.response_contract` 写入 `response.json` 后执行 `experiment-resume`。
command 方式调用受信任包装程序，协议不绑定模型供应商。

Ascend lower 路线链接原 [tilelang-ascend](../.agents/skills/tilelang-ascend/SKILL.md) 时默认按以下阶段推进：

```text
design → implementation → P1 全 case lowering → lowering_audit → validation
```

`skill-status` 定位下一块，`skill-read` 每次交付最多 40 行/2400 字节原文。
每步提交引用与带 SHA256、行号的证据；lowering 审计逐 case 引用本次 CAS 中的 TIR/ASC。
源码冻结时同时登记设备门禁，四阶段完成前不能预留该 trial 的设备作业。
每个阶段分别计入 generation/token；四步仍是一个独立试验，修复保留父子关系。

`CONTRACT_CHECKED` 仅表示交付、来源与结构检查通过，不能证明模型理解充分或技术结论正确。
显式 `skill_workflow: "links"` 保留只给链接的协议，适用于消融和旧接线检查。

## 验证和保证范围

当前可复现的框架回归覆盖契约、错误输出、输入修改、源码漂移、超时、恢复、配额、分阶段交接和审计门禁。
固定响应 fixture 验证协议接线，不计作模型生成成功率。已记录的具体验证见[实现状态](docs/implementation-status.zh-CN.md)。

lowering、vendor build、device correctness、measurement 四层分别报告。
自动生成只运行 host/lower，设备作业需要显式单 case、调用额度和设备选择；没有自动设备批跑或正式性能晋升。
配对 NPU event 测量入口与 profiler 的 kernel 侧计时不是同一种口径，不能混用。

普通进程隔离用于开发中的崩溃和超时处理，不是运行恶意 Python 的安全沙箱。
同用户候选仍拥有该账户的文件与网络权限；设备门禁只约束 harness 管理的作业。

## 维护固定源码身份

oracle 和基线源码的字节变化会使 fixture 失效，包含格式化产生的变化。审核变更后显式执行：

```bash
python harness/tools/freeze_fixtures.py --write
python harness/tools/freeze_fixtures.py
```

重新冻结会产生新的任务身份。旧实验使用对应的冻结 runtime 和 Skill 版本，不替换源码后续写历史结果。
