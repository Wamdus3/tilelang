# P1/P2 生成对比实验

控制器为确定性状态机。一个外部 agent runtime 通过文件或命令协议交付候选；
第一轮使用 implementer 上下文，修复轮使用 diagnoser/implementer 上下文，不增加统筹 Agent。
内置 vecadd/RMSNorm、SSDChunkStateFwd 四项任务、FFT complex64 pilot，以及 `fft_c2c_complex64` / `fft_c2c_complex128` 完整 workload 任务。
后两项分别包含指定 TileOPs 提交的 8 条 complex64 和 2 条 complex128 shape/dtype 配对，最大长度 1,048,576。
逻辑复数以独立实部/虚部 float32[B,N] 或 float64[B,N] 表示；来源、原始 batch 维与扁平映射保存在 `fixtures/fft_c2c.tileops.json`。
这是任务与源码生成覆盖；当前 dav-3510/Bisheng 不允许 aicore double 算术，complex128 厂商编译失败，不能解释成已支持运行。
各任务保持固定 case、oracle 和所有必需输出。

## 引用已有 Skill

对比配置直接链接 `.agents/skills/tilelang-ascend/SKILL.md`，不重写其指导，也不把内容复制进新 Skill。
初始化时递归解析其中的本地 Markdown 引用，记录仓库相对路径和 SHA256。
每轮请求的 `context.md` 提供链接，`context.json` 提供同一引用清单。
生成器应按任务读取这些原始文件。运行前及生成返回后检测引用漂移；发生漂移时停止，要求新建实验。
因此历史实验的指导文件需要由对应仓库版本保留；不能只靠修改后的同名路径重放旧上下文。

### 分阶段利用原 Skill（0.5.0）

新建 `lower` 实验且 arm 链接原 Ascend `SKILL.md` 时，默认启用 `skill_workflow: "ascend-kernel-v1"`。
原文件和引用闭包仍保持路径与哈希，不修改或另写算子指导。控制器按原文标题分配阅读范围，
每次 `skill-read` 只返回最多 40 行、2400 UTF-8 字节的原文，并在 SQLite/CAS 记录交付身份。
标题缺失、引用漂移或缺少结构/同步/测量参考会拒绝推进。不会把“计算文件哈希”计为 agent 阅读。

```text
design → implementation → P1 全 case lowering → lowering_audit → validation → 终态
```

| 阶段 | 使用原 Skill 的范围 | 必需的可检查交付物 |
| --- | --- | --- |
| design | 主技能工作流、kernel-structure、自动 pipeline 四步 | stage/core/PIPE/VF 表；含 padding、alias、转换 scratch、版本的物理预算；pipeline 上界与替代路线理由 |
| implementation | 主技能 API、结构/同步、build 和编译工具引用 | `solution.py`；选用 API 的 dtype/scope/地址对齐契约；初始化、读写跨度、owner 和 pipeline 依据 |
| lowering_audit | 工作流、pitfalls、同步与编译参考 | 对每个固定 case 审计实际存储、版本/epoch、ready/reuse、地址/跨度和 schedule；绑定本次 CAS 中的 lowered TIR/ASC |
| validation | 性能章节和 profiling | lower/vendor/device/performance 分层状态；覆盖、计时边界和次数；尚未完成的硬件验证与授权限制 |

每阶段使用独立 `requests/<trial>/<stage>/context.json` 和 `response.json`。同一生成 runtime
负责当前阶段，无需新增 Supervisor。前面阶段的文件在 `prior/<stage>/` 中，后两个阶段另有
`lowering/<case-id>/` 的准确产物副本；原始响应与产物哈希在 CAS 中保持不可变。

拿到 `WAITING_RESPONSE` 后，先读取该阶段 `context.md`，再按以下方式逐块读取：

```bash
$PY -m tilelang_harness skill-status --run-dir output/generation-study --trial-id "$TRIAL_ID"
# 将 next_chunk.chunk_id 赋给 CHUNK；每个工具返回只执行一次 skill-read：
$PY -m tilelang_harness skill-read --run-dir output/generation-study --trial-id "$TRIAL_ID" --chunk "$CHUNK"
```

保存每次返回的 `delivery_id`；如工具返回被截断，单独重读该块后再确认。`skill-status` 只返回
下一块和进度，避免为了找到未读部分而再次输出整份上下文。不能合并几十次读取到一个大输出。

响应格式以本次 `context.json.response_contract` 为准；四阶段响应还需要 `stage`、`request_id`、
`reads` 和 `checks`。每项 check 提供具体决定/偏离理由、原文短引用和证据的文件路径、SHA256、行号。
引用必须来自该项指定的参考文件；lower 审计必须覆盖每个 case 并引用其本次 `lowered.tir` 或
`kernel.cpp`，不能只提交一段泛泛说明、引用 input TIR、拿别的 case 或旧编译产物替代。
`NOT_APPLICABLE` 仍需具体范围理由及证据；`BLOCKED`/缺读/缺项/哈希错误均不能进入下一步。
实现阶段 API 契约应在 P1 lowering 前核实，不能等 vendor 编译后才补地址对齐审计。

每个阶段请求分别预留一次 generation 和 token，第一轮最低 4 次；lowering 只计一次 evaluation。
修复是一个新的 trial，重新走完整流程，不能用旧源码的审计为修复后源码背书。
报告保留每阶段 token/日志/原始响应和失败读取记录，不将四阶段计为四个独立算子样本。
budget 不足时拒绝初始化或记录耗尽，恢复不会重新请求已经提交的阶段。

实现源码冻结时，同一 SQLite 事务登记设备门禁；四阶段完成之前，相关 campaign 的设备作业
在资源预留前被拒绝。完成后仍要求显式 trial 关联、设备额度和单 case 命令，**不会自动执行
vendor 编译或 NPU 测试**。该门禁只约束 harness 管理的作业，不能限制外部脚本自行调用 SDK。

这建立的是**小块交付、明确确认、引用/证据来源与步骤顺序**，不能证明模型理解了原文，
也不自动判定技术解释正确。报告称为 `CONTRACT_CHECKED` 和 agent-reported evidence；
不能把这一状态当成 vendor/device/performance PASS，仍需技术审查和相应层级的实际验证。

需要复现实验旧协议或单独做“只给链接”的消融时，显式设置 `skill_workflow: "links"`；
不提供 skill 的 arm 仍使用原协议。它们不会被报告成完成四步 skill 流程。
旧实验保留原冻结 runtime，不通过替换当前源码续写历史成绩；本次变更需新建实验。

## 最小使用流程

以下命令在本地 WSL 的仓库根目录运行，不访问 A5，不运行 NPU：

```bash
export PYTHONPATH="$PWD/harness/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=.venv-wsl/bin/python
$PY -m tilelang_harness experiment-init harness/examples/generation-comparison.json \
  --repo-root "$PWD" --run-dir output/generation-study
$PY -m tilelang_harness experiment-run --run-dir output/generation-study
```

模板使用 `file` 生成器。返回 `WAITING_RESPONSE` 和一个 `request_directory`；
将该目录的 `context.md`、`context.json` 交给实际模型/agent，原子写入该目录的 `response.json`，再执行：

```bash
$PY -m tilelang_harness experiment-resume --run-dir output/generation-study
$PY -m tilelang_harness experiment-report --run-dir output/generation-study \
  --output output/generation-study/report.json
```

设置真正的模型身份、参数和独立重复次数后再开展对比。模板的模型名是待配置占位符。
当前模板演示单次直接生成与四阶段 workflow，二者调用结构不同，不能据此单独归因 Skill 的收益。
需要测量 Skill 本身的效果时，固定模型、调用结构和总 token/评测预算，仅改变指导变量；
报告总调用数、总 token、成功率与 kernel 耗时，不能只比较最后一次响应的成本。
每个 task/arm/replica 构成一个独立初始试验。修复轮保留 parent/root trial，继承任务和基点，
拿到上一轮错误分类、case 诊断、候选及评测目录。初始试验先排队，修复轮追加到队尾。
所有失败、未执行项和耗尽预算的项都保留；`COMPLETE` 只表示所有计划项已进入终态。

`links` 模式的响应为严格 JSON；分阶段模式使用前述扩展合同：

```json
{
  "schema_version": 1,
  "files": {"solution.py": "完整源码字符串"},
  "config": {},
  "plan": "实现假设及必要说明",
  "usage": {"input_tokens": null, "output_tokens": null}
}
```

源码入口由控制器固定：lower 模式为 `solution.py:build`，接收 case 的维度参数，RMSNorm 还接收 epsilon；
FFT pilot 接收 B,N；完整 workload 两任务接收 B,N,dtype，dtype 由任务合同提供，禁止用候选 config 覆盖。
host 模式为 NumPy 诊断 ABI `solution.py:run(inputs, epsilon)`。其他源码依赖一起放入 `files`。
生成器不能提供自己的 `candidate.json`、更改任务维度、路径逃逸、覆盖 oracle 或写入正式成绩。
这里的普通进程隔离仍不是恶意代码安全沙箱；同账户的恶意程序不在该保证内。

### FFT 全部 workload 的本地入口

```bash
$PY -m tilelang_harness validate fft_c2c_complex64
$PY -m tilelang_harness validate fft_c2c_complex128
$PY -m tilelang_harness selfcheck fft_c2c_complex64 --output output/fft-host64.json
$PY -m tilelang_harness selfcheck fft_c2c_complex128 --output output/fft-host128.json
```

新建 experiment 时将 tasks 设置为这两个名称；保留 `fft_c2c` 可复用原 pilot 的 8 个测试形状。
complex64 容差为 rtol=atol=1e-4，complex128 为 1e-10；这是本地实验预设，原 YAML 未指定容差。
两个完整任务的总元素上限为 4,194,304，其余 P0 任务仍保持百万元素上限。
CPU radix-2 基线按行分块限制临时存储，FP64 输入和输出全程保留精度，不用 FP32 近似替代。

P1 生成候选可用固定 UB、输出 GM 分阶段状态覆盖长 FFT，仍返回一个 PrimFunc；源码 lowering 与本地 `bisheng -c` 分开记录。
当前固定设备种子只覆盖 N<=4096，长 FFT 的正式配对测量基线尚未提供；该路径明确报不可用，不进行静默降级。
FP64 生成 ASC 的浮点常量还存在带 `f` 后缀后转 double 的精度风险，解除厂商编译禁令后也需先修正常量输出并验证。
shape 与来源以随包发布的 fixture 和 scope 清单为准；本地原始报告不随源码发布。

## 接入命令式 runtime

将 generator 改为 `kind: "command"`，`command` 为受信任包装程序的 argv 数组。
控制器追加 `--request <context.json绝对路径> --response <response.json绝对路径>`，不使用 shell 字符串拼接。
包装程序负责读取请求中的 model、parameters、seed、token_reservation、Skill 链接及反馈，
调用所选 runtime，并将结果转换为上述响应。协议不绑定模型供应商，也不要求安装特定模型 SDK。

命令、模型参数、可识别的命令文件内容、提示上下文、原始响应、stdout/stderr 和结束原因均记录。
不要把 API key 等凭据写入 command 参数、model parameters 或响应；由 runtime 自己的凭据机制处理。
源码相同的候选可按 CandidateID 去重，但生成 TrialID 和评测 AttemptID 独立，不能据缓存虚增模型成功次数。

每轮调用前预留一次生成额度和 token 上限；失败及中断也计入，未知 token 用量保持 unknown。
上游 runtime 必须实际执行 token 限额：本框架能控制是否再次调用，不能撤销外部服务已经发生的超额消耗。
若上游报告超出预留的 token 用量，记录失败并阻止后续生成。
命令有独立 watchdog，控制器退出后仍受 deadline 控制；恢复时对已释放请求只收集结果，不重复发起，
之后才推进尚未开始的 trial，预算不会重置。`generations` 计适配器请求次数；
若 runtime 内部有多轮模型调用，需由 runtime 自行限制并汇总 token 使用，不能把它自动等同为一次模型 API 请求。
文件交接等待时间计入整个实验墙钟预算，避免无限挂起。

## 报告和公平对比

报告按 task/arm 汇总计划数、首次成功率、预算内成功率、生成预留次数、唯一候选数、失败分类、
已知 token 用量、未知用量项和可获得的生成墙钟时间；保留各次原始响应和评测证据。
首次成功率分母为预先计划的初始试验数，修复成功不回填到首次成功。
预算内成功按 root trial 计一次，重复生成同一份源码不会被当成多个独立优胜候选。

`evaluation_mode=lower` 的成功仅表示源码 lowering 成功，绝不能标作设备正确率。
host 成功仅为 NumPy 诊断；没有任何路径凭这些结果授予正式性能晋升。
模型、Skill、修复预算等变量分别控制；例如比较 Skill 时两组使用相同模型、参数、任务与额度。
完整 workload 的设备分数、自动优胜晋升、Autotune、Macro/Recipe 不在本次生成对比版范围内。

## P1 单 case 设备验收与调用预算

自动生成实验只允许 host/lower。设备操作继续使用显式单 case 命令；A5 不提供批跑入口。
新 campaign 的设备额度默认 0；需要设备时显式设置总额度，再为单个作业预留调用上限。
框架的数值检查会调用三次：原输入、原对象原地改变输入、新输入对象。
因此正确性模式的最小预算是三次，测量还需要基线及测前测后检查与计时调用。

下面是待授权设备操作的命令格式，不属于上方自动实验：

```bash
# RUN_DIR 是相应 task 的 campaign；CID/TRIAL_ID 来自 experiment-report。
python -m tilelang_harness device-quota --run-dir "$RUN_DIR" --launches 3
python -m tilelang_harness run "$CID" --run-dir "$RUN_DIR" --mode device-smoke \
  --case n257_normal --device-index 0 --max-launches 3 --trial-id "$TRIAL_ID"
```

显式 trial 关联的设备结果会单独列入实验报告，保留 case 覆盖；不改写原有 lowering 成功率。
同一用户设备租约、健康/零利用率检查、编译后再次检查、进程 deadline 与异常隔离继续生效。
有驻留进程时必须另外显式选择 `--allow-resident-processes`，依然要求健康且实时利用率为 0%。

配额按框架调度的 TileLang callable 调用计数，在调用前持久化，失败/未知完成也消耗额度；
按作业预留上限计入 campaign，未使用额度不自动返还，以免中断恢复时重复启动。
这不限制恶意候选自行发起设备操作，也不声称计入第三方 profiler 的辅助 kernel。
默认不启用计时；使用 `--measure` 时需明确给足额外额度，触顶会失败而不会伪造完整样本。
计时实现使用单次 callable 的 NPU event，配额持久化发生在 event 开始之前；不引入 profiler 的缓存冲刷辅助 kernel。
warmup/repeat 毫秒字段是本实现的主机墙钟预算，保存每次实际调用的 event 时长和 warmup 次数。
该计时路径只完成主机边界测试，尚未取得真机性能样本，不能据此报告速度或晋升。

## 验证范围

P2 测试中的生成器是明确标注的协议 fixture，只证明接线、恢复、预算和诊断逻辑，不是模型生成能力结果。
真实 lower 集成记录同样使用固定基线响应，验证统一流水而非模型优劣。
真实模型对比应在配置 runtime 后产生新实验，不得把这些 fixture 成绩计入模型报告。
