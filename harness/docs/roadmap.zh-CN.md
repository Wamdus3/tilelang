# 后续计划

本文件区分 0.5.0 的已实现基础与后续设计。当前实现见[状态表](implementation-status.zh-CN.md)，
后续工作继续由确定性控制器承接，不引入常驻统筹 agent。

## 近期：建立可信的生成对比

- 配置真实模型/runtime，冻结模型参数、Skill 版本、输入、case、精度和全部调用成本。
- 分别比较单次生成、只给 Skill 链接、分阶段 Skill；比较 Skill 本身时控制调用结构与总预算。
- 对阶段决策做技术审查，补足“证据结构正确但内容错误”的诊断样本。
- 保留 direct 作为实验对照，报告独立样本数、首次成功率、修复成本和最终 kernel 耗时。
- 将完整 workload 的设备正确性及 profiler kernel 计时形成统一、可归档的评测 adapter。

已有 NPU event callable 测量入口不能直接替代 profiler kernel 计时。
模板演示的四阶段与单次调用预算不同，不能据此将结果单独归因于 Skill 内容。

## 中期：搜索与诊断

- 以算法结构、参数搜索和 shape 特化逐层扩展，保留候选父子关系和不可变历史最好。
- 接入受控 Autotune，记录配置、实际编译数、失败、耗时及胜出配置复核。
- 补充面向算子族的 TaskAdapter、多 kernel callable/dispatch 合同和独立设备标杆。
- 接入 profiling、lowering、版本/容量与同步诊断，基于证据选择下一项最小实验。
- 在明确 noise、覆盖和回归策略后实现 PromotionPolicy；当前不自动授予正式晋升。

## 后期：Macro、知识和编译器共同演进

- 将已验证阶段抽成带调用合同、布局、物理存储和测试的 Macro/Recipe。
- KnowledgeEngine 使用适用条件与反例选择方法，经验先进入待审核区，不热改当前实验 Skill。
- 编译器改进走独立补丁/回归路径，以新 CompilerPack 开启新实验，避免混入原 trial。
- 为跨机器 transport、任务队列及恶意代码隔离增加独立边界，不扩大现有进程隔离的保证。

这些项目尚未作为当前框架能力交付，原有架构草稿中的检索、知识晋升、Macro 库和自动搜索
不应仅因存在同名设计对象就标记完成。
