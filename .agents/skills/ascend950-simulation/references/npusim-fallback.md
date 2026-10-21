# npusim 备用路径

用于微基准、原始 SQLite 查询、VF IPC 或与 msprof 独立交叉验证。
默认 Insight 指令详情交付按主 SKILL 使用 msprof。

```bash
npusim record -s Ascend950 -o ./sim-new -g -n all /absolute/path/to/acl_launcher
# Python/JIT 使用已验证解释器；引号内的完整调用按当前 npusim 帮助确定。
npusim record -s Ascend950 -o ./sim-python-new -g -n all \
  '/absolute/path/to/python /absolute/path/to/kernel_launcher.py'
```

`-n` 控制日志核范围，launch 核数由 launcher 决定；只有 core 0 时明确局部范围。
已验证 CLI 的 `-g vf` 额外导出 VF IPC JSON。需要重生成报告时复用原 archive：

```bash
npusim report -e /path/to/npusim_ARCHIVE -o ./new-pipeline-report -n all \
  --smart-pipes --merge-trace
```

检查实际单核报告与 `trace_allchips.json`，顶层 index 警告不能代替文件检查。
原生报告曾在 SIMT 用例漏画 SQLite 已记录的 RVEC 区间；必要时从 SQLite 导出：

```bash
mkdir -p ./new-sqlite-detail
python /path/to/ascend950-simulation/scripts/export_pipe_trace.py \
  /path/to/npusim_ARCHIVE/record/kernel.db ./new-sqlite-detail/trace.json --layout pipe
```

- 已验证 CANN 9.2.0 的 `ExecutedInstructions` 含 start/end tick，时钟来自
  `KernelInfo.frequency_ghz`。旧导出器按 core/硬件 PIPE 共轨，将 RVEC 子类型合到
  VECTOR；`--layout slots` 只是虚拟显示分槽，`--overview` 仅按需另存区间并集。
- helper `status=ok` 只证明写出成功，检查 `start_overwrites`、`end_overwrites`、
  `start_only`、`end_only`、`mismatched`、`invalid`、`write_failed`。
  SIMD/mixed 曾出现覆盖和未配对，SQLite 导出不能恢复缺失事件。
- `end_only` 应逐条定位。只有证明为零耗时 GET/RLS 后，才与计算/搬运缺口区分；
  保留原始配对指标，不把“数量恰好等于 GET/RLS 数”当作身份验证。
- 已验证 helper 0.8.0 将同步记录归一化为 1 tick，包括 WAIT_FLAG 和 intra-block；
  不能由此读取真实等待区间。msprof 的 ResourceConflictRatio 区间与此语义不同。
- 旧 SQLite 导出器的 intra-block 仍为 X slice，不能据 JSON 配对数宣称旧 Insight
  可点击关联；需要完整等待区间和经验证连线时使用主流程的 msprof 合并版。
- 若出现 `instr.bin`/`chip*_instr.bin` 而非 SQLite，使用同版 decoder，先验证字段
  和单位。已验证记录 `is_popped=1` 为 dispatch、0 为 completion，更换 reader 后重验。
- 核对目标身份、输入输出、launch/chip/core、ID/PC 和来源，独立运行间不要对齐或
  拼接绝对 tick。正确性、采集完整性、仿真周期和真机验证状态分别报告。
