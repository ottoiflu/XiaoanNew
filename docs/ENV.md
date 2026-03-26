# XiaoanNew 纯 uv 环境管理规范 (v2.1.2)

## 1. 核心设计哲学：全自动闭环
本项目摒弃传统的 Conda/Pip 混合模式，采用 **“项目级环境隔离”** 策略，利用 `uv` 实现极速、稳定、可复现的深度学习开发体验。

*   **单一入口**：`pyproject.toml` 定义所有业务逻辑依赖与源路径。
*   **绝对锁定**：`uv.lock` 替代 `requirements.txt`，确保环境在不同机器间 **100% 字节级一致**。
*   **零配置运行**：通过 `uv run` 实现“即装即用”，无需手动维护环境变量。
*   **硬链接存储**：利用 `uv` 的全局缓存机制，相同版本的包在磁盘上仅存一份，极大节省云服务器存储成本。

---

## 2. 环境初始化 (新设备/新成员)

### 第一步：安装 uv 引擎
项目不再依赖 Conda。若宿主机未安装 `uv`，请执行：
*   **Linux/macOS**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
*   **Windows**: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

### 第二步：一键构建环境
进入项目根目录，直接执行以下命令：
```bash
# 1. 锁定 Python 版本 (若系统缺失，uv 会自动下载隔离的二进制版)
uv python pin 3.10

# 2. 一键创建 .venv 并安装所有依赖 (包括 Torch CUDA 专版)
uv sync
```
> **注**：由于 PyTorch 的 NVIDIA 依赖包体积较大（约 2GB+），首次同步需 2-5 分钟。`uv` 会自动处理并发下载，后续同步将实现“秒级完成”。

---

## 3. 日常开发工作流

### 3.1 依赖管理 (全自动化同步)
**严禁手动编辑 `pyproject.toml` 的 dependencies 字段**，请通过 `uv` 指令保持配置文件与环境的同步：

| 需求 | 命令 | 效果 |
| :--- | :--- | :--- |
| **增加运行时依赖** | `uv add pandas` | 自动下载 + 写入 toml + 更新 uv.lock |
| **增加开发工具** | `uv add --dev ruff` | 归类至 `[project.optional-dependencies]` |
| **移除依赖** | `uv remove opencv-python` | 物理清理包并同步更新配置文件 |
| **同步最新配置** | `uv sync` | 当你 `git pull` 后发现依赖变动时执行 |

### 3.2 运行程序与脚本
无需执行 `source .venv/activate`。使用 `uv run` 运行可确保程序永远处于正确的虚拟环境中。
*   **运行主程序**：`uv run python app.py`
*   **运行工具脚本**：`uv run python scripts/contrast_test_v2.py`
*   **执行测试**：`uv run pytest`

---

## 4. PyTorch CUDA 专项配置规范
为确保 `uv` 能够精准命中 NVIDIA 加速版本，并避开普通 PyPI 源中的“全家桶”冲突，配置文件已固定以下 **“专包专线”** 逻辑：

```toml
# 1. 设置默认主源 (加速 90% 的普通第三方包)
[[tool.uv.index]]
name = "tsinghua"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true

# 2. 定义 PyTorch 专用 CUDA 源 (根据系统显卡驱动选择 cu118 或 cu121)
[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true  # 强制隔离：除非明确指定，否则普通包不会去此源寻找

# 3. 建立包与源的绑定关系
[tool.uv.sources]
torch = { index = "pytorch-cu118" }
torchvision = { index = "pytorch-cu118" }
torchaudio = { index = "pytorch-cu118" }
```

---

## 5. 系统级依赖说明 (System Dependencies)
`uv` 负责管理 Python 及其依赖包，但某些库（如 `opencv-python`）需要 Linux 系统底层图形库的支持。

若运行报错 `ImportError: libGL.so.1: cannot open shared object file`，请在 **宿主机** 执行：
```bash
sudo apt-get update && sudo apt-get install libgl1 libglib2.0-0 -y
```
*在云平台（AutoDL/SeetaCloud）中，建议优先考虑安装 `opencv-python-headless` 以规避系统图形库依赖。*

---

## 6. 进阶维护指令
*   **全量升级**：`uv lock --upgrade`（将所有依赖升级到 `pyproject.toml` 允许的最新版本）。
*   **查看依赖树**：`uv tree`（排查包冲突利器）。
*   **查看磁盘占用**：`uv cache clean`（清理缓存）或直接查看 `.venv`。
*   **环境重建**：若环境损坏，直接 `rm -rf .venv` 然后重新 `uv sync`。

---

## 7. Git 协作守则
为了保证团队/多端协作不冲突，请务必遵守以下规则：
1.  **必须提交**：`pyproject.toml` (定义依赖规则) 和 `uv.lock` (定义确切版本)。
2.  **绝对禁止提交**：`.venv/` (虚拟环境实体) 和 `__pycache__/` (编译缓存)。
3.  **冲突处理**：若 `uv.lock` 产生 Git 冲突，通常只需执行 `uv lock` 即可重新生成正确的锁文件。

---