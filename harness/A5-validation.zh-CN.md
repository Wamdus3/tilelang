# 设备验证说明

本框架的设备入口为显式单 case `device-smoke`，自动生成实验仅允许 host/lower。
首次接入机器前先检查环境、设备型号与编号映射，不通过环境探测自动启动算子或性能采样。
服务器上的实际作业需单独获得授权，提前列明命令和调用预算；TileLang 源码构建在本地 WSL 完成。

## 环境和设备检查

`tools/probe_npu_environment.py` 只读检查 Python/库版本、设备状态和依赖。
离线环境方案见[环境准备](environments/a5/README.zh-CN.md)。检查 Python、native 库、CANN、
Bisheng、torch_npu 的实际来源，不能仅依赖环境名称。记录逻辑设备索引对应的物理卡。

设备作业要求健康且实时利用率为零。默认拒绝驻留进程；只有已获授权时才显式使用
`--allow-resident-processes`，并在构建前、启动前重复检查。

## 调用配额

新 campaign 默认设备额度为零。一次正确性作业执行原输入、原地改变输入和新对象输入三种调用，
最小预算为三次；失败和未知完成也消耗额度。限额在调度和每次调用前持久化，不自动返还。

以下只是授权后的命令模板，变量来自已冻结 campaign/experiment：

```bash
python -m tilelang_harness device-quota --run-dir "$RUN_DIR" --launches 3
python -m tilelang_harness run "$CID" --run-dir "$RUN_DIR" --mode device-smoke \
  --case "$CASE_ID" --device-index "$DEVICE_INDEX" --max-launches 3 --trial-id "$TRIAL_ID"
```

分阶段 Skill 生成的候选须先完成审计门禁；完成门禁不授予设备权限或调用额度。
同一源码若来自多个 trial，必须使用当前路线的明确关联，不能借用另一条路线的审计状态。

`tools/a5_vecadd_three_runs.py`、`tools/a5_flash_attention_smoke.py` 与
`tools/check_p1_device_campaign.py` 是受控验证入口，运行前必须核对其源码、参数和调用次数。
它们不代表允许批跑或调优，也不自动访问某个固定服务器。

## 测量和证据

不带 `--measure` 的作业只做正确性检查。测量需要额外基线、测前/测后检查及计时调用预算。
当前 `evaluation.py` 提供 NPU event 的完整 callable 配对测量，默认任务协议记录原始样本与单位。
用户要求的 profiler kernel 侧耗时需要对应 profiler adapter，不与 event 计时混为同一成绩。

报告分别保存 lowering、vendor build、device correctness 和 measurement。
单 case 通过不能回填完整 workload 成功率，也不能将以前的 wheel/运行组合视为当前版本已验证。

历史记录中，固定 vecadd 的 P1 单 case 曾完成一次 vendor 构建和三次正确性调用，measurement 未运行。
这是旧快照的有限证据；0.5.0 Skill 状态流及后续整理版未据此宣称新的设备或性能验收。
原始日志、数据库、机器地址和个人前缀均保留在本地，不属于源码交付。
