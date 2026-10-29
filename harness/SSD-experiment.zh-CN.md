# SSDChunkStateFwd 接口

固定来源是 TileOPs 提交 `b84f999c9ec5d375bcedb62b44d79863f830924a` 的
`src/tileops/manifest/spec/mamba.yaml` 中六条 `SSDChunkStateFwdOp` workload。
四个任务按 float16/bfloat16 和 plain/seq 分组，case ID 为 `w1_fp16_plain` 至 `w6_fp16_seq`，
完整 shape、种子及参考身份保存在随包发布的 `src/tilelang_harness/fixtures/ssd_chunk_state_fwd_*.json`。

生成入口为 `build(B,NC,Q,H,P,G,N,dtype,dt_dtype,has_seq_idx) -> PrimFunc`；
参数顺序为 `x[B,NC*Q,H,P]`、`Bmat[B,NC*Q,G,N]`、`dt[B,H,NC,Q]`、
`dA[B,H,NC,Q]`、`seq_idx[B,NC*Q]`、`states[B,NC,H,P,N]`。
`x/Bmat` 使用任务 dtype，`dt/dA/states` 为 FP32，`seq_idx` 为 int32。
plain 模式仍传真实 int32 张量，kernel 和通过任务 dispatch 取得的 host 参考均忽略其内容。
`oracle_for` / `host_baseline_for` 根据任务绑定 `has_seq_idx`，host seed 也保存该编译期标志；
plain 模式不依赖 dummy seq 恰好为零。

归约使用 `exp(min(dA_end-dA_q,0))*dt_q*x_q*B_q`，head 映射为 `h//(H/G)`。
seq 模式仅累加与 chunk 最后 token 相同的序列；最后 token 序列为负时整个 chunk 为零。
精度沿用上游 workload verification：rtol=1e-3，FP16 atol=1e-3，BF16 atol=1.6e-2。
CPU oracle 用 FP64 einsum，独立 host baseline 用 FP32 matmul。
离线 BF16 输入在 NumPy FP32 数组中承载精确 BF16 值，采用 round-to-nearest-even；
这项主机表示适配不代表 NumPy BF16 或 NPU 正确性已验证。

AscendC 标杆由外部统一评测取得，内置设备 baseline 明确抛出 unavailable，避免把生成候选当成标杆。
`seed --abi numpy` 支持主机诊断。生成实验使用真实 file 请求，并记录每个任务独立 trial；
同一个源码覆盖四个任务时只算一次独立生成，不能将任务拆分计成四次模型成功。

运行命令：

```bash
PYTHONPATH=harness/src:. .venv-wsl/bin/python -m tilelang_harness validate ssd_chunk_state_fwd_float16_plain
PYTHONPATH=harness/src:. OPENBLAS_NUM_THREADS=2 .venv-wsl/bin/python -m tilelang_harness selfcheck ssd_chunk_state_fwd_float16_plain
PYTHONPATH=harness/src:. .venv-wsl/bin/python -m pytest harness/tests/test_ssd_chunk_state.py -q
```
