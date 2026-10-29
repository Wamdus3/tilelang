# 本地准备与离线部署 Ascend 环境

工具先在本地 WSL 按本仓库依赖准备独立环境，再按明确前缀离线部署。
不克隆共享环境，不在服务器构建 TileLang，不修改驱动、固件或共享 CANN。
以下为可配置的操作流程，版本文件是已记录的环境快照，不代表自动跟踪最新配套标准。

## 工具和材料

| 文件 | 用途 |
| --- | --- |
| `runtime.in` / `build.in` | 固定运行依赖和本地构建工具版本 |
| `prepare_local.py` | 校验仓库根依赖、下载 wheel 闭包和生成哈希清单 |
| `prepare_cann920.py` | 下载 CANN 9.2.0 材料并记录来源、大小与哈希 |
| `audit_abi.py` | 检查实际 ELF 所需 GLIBC 版本，拒绝超限与 GLIBC_PRIVATE |
| `build_local.sh` | 限定本地 WSL，用独立编译链构建 TileLang 与 harness wheel |
| `ffi_portable_options.cmake` | 本地构建的 TVM FFI 配置 |
| `pack_local.py` | 安装、导入验证、ELF 审计和 conda-pack 归档 |
| `install_offline.py` | 校验归档、拒绝路径逃逸/覆盖，并执行 conda-unpack |
| `verify_local.py` / `verify_imports.py` | 本地依赖及模块来源检查 |

已记录的部署组合为 Linux x86_64、目标 GLIBC 2.31、Python 3.11、Torch 2.10.0+cpu、
 torch_npu 2.10.0.post4；后续补充 CANN 9.2.0。是否适用于新的驱动/设备组合需要独立验证。
不要用缺少本仓 Ascend 方言的通用 PyPI TileLang wheel 替代本地构建产物。

## 本地 WSL 流程

从仓库根目录执行，使用本机现有的 Python、micromamba 和已配置编译链。
所有网络下载、依赖求解及源码构建均在本地进行：

```bash
export TLH_A5_ARTIFACT_DIR="$PWD/output/a5-offline"
export TLH_A5_LOCAL_WORK="$HOME/.cache/tilelang-950/a5-offline"
python harness/environments/a5/prepare_local.py --output "$TLH_A5_ARTIFACT_DIR"
python harness/environments/a5/audit_abi.py "$TLH_A5_ARTIFACT_DIR/wheels" \
  --max-glibc 2.31 --output "$TLH_A5_ARTIFACT_DIR/wheel-abi.json"
# 准备 build-toolchain 和 target-env 后，在独立工具链环境执行：
micromamba run --prefix "$TLH_A5_LOCAL_WORK/build-toolchain" \
  bash harness/environments/a5/build_local.sh
python harness/environments/a5/pack_local.py \
  --output "$TLH_A5_ARTIFACT_DIR" --prefix "$TLH_A5_LOCAL_WORK/target-env" \
  --packing-python "$TLH_A5_LOCAL_WORK/runtime-env/bin/python"
```

`TLH_A5_LOCAL_WORK` 默认位于本机用户缓存，可通过环境变量调整。
先检查各脚本 `--help` 和清单要求，基础 Conda 运行库、sysroot 及打包工具环境由操作者准备。
只有完整审计成功并写出 `READY_FOR_OFFLINE_INSTALL` 后才部署 bundle。

## 获得授权后的部署

仅传输 bundle 中的归档、清单和安装器到指定个人目录。
安装器强制显式提供目标 workspace，防止默认为某个账户路径：

```bash
python3 harness/environments/a5/install_offline.py \
  --bundle "$BUNDLE_DIR" --workspace "$TARGET_WORKSPACE"
```

系统 Python 3.8 可解析安装器。安装不覆盖已有前缀、不修改 `.bashrc`；失败保留未完成标志，
先核对失败原因，不能对不明状态反复解包。环境激活应明确 CANN 来源和个人运行库前缀。

导入检查使用 `python -I -m pip check`，并显式导入 torch_npu，再检查 TileLang Ascend 方言。
版本、依赖及导入通过不代表设备数值或性能通过；设备入口见[验证说明](../../A5-validation.zh-CN.md)。
归档、包缓存、安装日志、服务器地址和个人前缀不随框架源码发布。

## 本地工具测试

```bash
python -m pytest harness/environments/a5/tests -q
```
