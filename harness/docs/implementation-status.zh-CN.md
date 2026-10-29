# 实现状态与验证

版本：0.5.0。整理日期：2026-10-29。本文件描述实际交付，不将架构草稿或外部实验脚本计作内置能力。

## 已实现

| 阶段 | 实现内容 |
| --- | --- |
| P0 | 严格任务/来源合同、独立 oracle 与基线、固定输入和覆盖、完整输出及输入生命周期检查 |
| P1 | bundle 冻结、SQLite/CAS、身份规约、独立 worker/watchdog、硬超时、恢复、重放和分层报告 |
| P1 设备接口 | 同用户租约、健康/利用率检查、显式单 case、持久配额、同 handle 正确性及 NPU event 测量入口 |
| P2 | file/command 生成协议、独立 trial、模型/参数记录、有限修复、全部调用成本和分组统计 |
| P2 Skill | 原文件引用闭包、分块交付、四阶段证据合同、逐 case lowering 审计及 candidate/trial 设备门禁 |
| 任务 | vecadd、RMSNorm、FFT pilot/完整 FP32/FP64 合同、SSDChunkStateFwd 四项任务 |
| 环境辅助 | 本地依赖闭包、ELF ABI 审计、离线打包与显式前缀安装 |

Ascend 源码 lowering 导出真实 TIR/ASC；当前 FP64 workload 只有主机参考和 lowering 合同，
已记录的 dav-3510/Bisheng 不支持 aicore double 算术，不能据此宣称 complex128 设备执行支持。
SSD 设备 baseline 明确 unavailable；外部 AscendC 评测不能自动变成内置 baseline。

## 验证范围

基于最新 `origin/main` 拆分后的 0.5.0 快照在本地 WSL 完成框架主机测试 **167 passed / 1 skipped**、
环境工具 **13 passed**、模板主机测试 **3 passed**，合计 **183 passed / 1 skipped**。
限定变更文件的 `format.sh --files` 检查通过。
常规跳过项是真实 Ascend lowering 的显式集成测试。基于新主线在本地 WSL 独立重建编译库，
检查实际 worker 的 CompilerPack/动态库来源后，另行开启该项通过：
固定 vecadd 基线、4 次阶段交接、1 次评测、7 个 case 的真实源码 lowering。
这些是框架 fixture，不是 LLM 生成成功率；vendor/device/performance 均未在本次修复中执行。
超时/后代清理 fixture 使用 10 秒预算以留出进程启动时间，仍断言超时和全部后代终止；
对应 P2 文件串行复核为 **23 passed**。

独立 wheel 在仓库外完成全任务 selfcheck，以及 vecadd/RMSNorm 的主机 campaign 和重放。
9 项任务的 oracle/基线源码身份为 MATCH；格式化后显式重新冻结，算法 AST 保持不变。
原始运行日志和机器路径留在本地，当前 commit 对应的验证命令见下方及提交记录。
历史设备单 case 的有限结果见[设备说明](../A5-validation.zh-CN.md)，不外推到当前快照。

### 可复现命令

```bash
python -m pytest harness/tests harness/environments/a5/tests testing/cpp/test_ascend_header_compatibility.py -q
python -m ruff check harness
python harness/tools/freeze_fixtures.py
```

只在具备本地 Ascend 编译环境时显式开启真实源码 lowering 集成，不访问设备：

```bash
TLH_RUN_SKILL_LOWERING=1 python -m pytest \
  harness/tests/test_skill_workflow.py::test_original_skill_with_real_lowering_artifact_handoff -q
```

`check_campaign_lowering.py` 和 `check_experiment_lowering.py` 还可验证固定响应的统一链路及重放，
需要不存在的新输出目录。构建与编译库来源须在本地环境中确认，不能把 stub host 检查视为设备构建。

## 当前边界

- Skill 检查验证交付、来源、引用和结构，技术解释仍需审查，不能证明模型理解。
- 自动生成只执行 host/lower；设备权限、调用次数和实际作业分别控制。
- 完整 workload profiler kernel 计时、统一自动胜出晋升和正式性能分数尚未接入。
- GEMM/Attention TaskAdapter、Autotune、Macro/Recipe、知识检索和编译器闭环属于[后续计划](roadmap.zh-CN.md)。
- 进程隔离不是恶意代码安全沙箱；远程 transport、跨账户隔离和调度服务未实现。

## Ascend 模板依赖

模板兼容性修复单独提交，框架本体的主机状态机不依赖这些改动；使用对应 Ascend SDK 编译时需要这些修复：

- `numeric_limits.h` 使用 Bisheng 支持的 constexpr bit cast，兼容较旧宿主 C++ 标准库。
- `simd_inst.h` 修正 predicate pack/unpack 调用及 quarter pack store 的元素偏移。
- `debug.h` 对 block index 做显式整数转换，匹配调试输出的格式要求。

`testing/cpp/test_ascend_header_compatibility.py` 使用主机编译器与 SDK 形状 stub 检查位模式和包装分派。
这类测试不证明 NPU 指令的数值行为，亦不等同于完整 SDK 构建验证。
