# Insight 兼容与验证

## 事件与连线

- 使用能解析原名 intra-block B/E 的 Insight。26.2.0 已验证支持；不支持时报告版本
  限制，不将 intra-block 改写为 SET/WAIT，也不添加副本或显示轨道映射。
- GET/RLS X 的顶层 id 在已验证的 26.2.0 中仍不填充 slice.flag_id，因此原名事件
  保留在硬件 PIPE，另加带 `viewer_compatibility_anchor=true` 和来源信息的 B/E
  SET_FLAG/WAIT_FLAG 锚点。锚点不是实际执行的 flag 指令。
- GET/RLS 与锚点都从 raw completion 起显示 1 tick，语义耗时为零。锚点分别位于
  `PIPE / GET_BUF (compat)` 与 `PIPE / RLS_BUF (compat)`；s/t/f 端点使用对应锚点
  的 pid/tid 和起点。连线表示 acquire/release 范围，不代表指令延迟或跨 PIPE 依赖。
- 分离两端泳道是为了避免已观察到的 `(flag_id, track_id)` 首条 slice depth 查询
  忽略端点时间，以及同轨长区间遮挡。仅修改 thread_name 不能改变归属，须核对 pid/tid。
- 色块和连线使用 sunsetOrange（调色板索引 9）。色块按 cname 哈希选色，线按 cat
  哈希选色；脚本的 `GET/RLS buffer scopes` 对应同一橙色，仅改 flow.cname 无效。

## 实际验证

使用新目录导入，避免旧解析数据库缓存；区分以下验证层级：

1. JSON/raw：源记录覆盖、PC/tick/core/PIPE、配对 ID、真实指令与锚点分别计数。
2. 解析库：GET/RLS 原名、intra-block 原生区间、锚点 flag_id 与 flow 的关联。
3. 流水接口：`unit/threadTraces` 与 `unit/flows` 逐端点比较时间、泳道和 depth；
   `flow/categoryEvents` 另验全量连线。单节点成功不证明全量视图正确。
4. GUI：实际点击色块与检查连线。未测试时明确说明，不将接口验证称为 GUI 验收。

1 tick 在 1.65 GHz 下约 0.606 ns，整数纳秒解析可能量化为 0 ns，影响可见性/命中。
保留 1 tick 并报告实际限制，不静默扩大宽度。空查询结果需检查版本、请求参数和数据，
不能直接判定无事件，也不能沿用旧版本的验证结论或将 GEMM 结果外推至 VF/MHA。

点击 GET/RLS 的兼容 SET/WAIT 锚点查连线，不能宣称原名 X 已支持直接关联。
若用户选择修改解析器，可参考 [X.id 补丁](insight-x-flow-id.patch) 并检查目标版本；
未安装/验证时不声称生效，手改解析数据库会被重新导入覆盖。
