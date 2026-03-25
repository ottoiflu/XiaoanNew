# XiaoanNew 环境管理规范文档

> 版本 2.1.2 | 最后更新 2026-03-25 | 基于实际部署环境审计

---

## 1. 当前环境快照

| 项目 | 值 |
|------|-----|
| Conda 环境名 | `XiaoanNew` |
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| CUDA (PyTorch) | 12.1 |
| CUDA (Driver) | 12.2 |
| GPU Driver | NVIDIA 535.154.05 |
| uv | 0.10.12 |
| Conda | 24.11.3 |
| 已安装包数 | 103 |
| 可编辑安装 | 已启用 (egg-info) |

---

## 2. 核心设计：分层依赖管理

本项目采用三层分离的依赖管理策略，每层文件职责不同，严禁混用。

```
environment.yml      Conda 层：只管 Python 版本 + uv 工具
        |
        v
pyproject.toml       项目层：声明依赖范围、构建配置、开发工具
        |
        v
requirements.txt     锁定层：由 uv 自动生成的精确版本快照
```

### 2.1 environment.yml（Conda 基础外壳）

仅负责两件事：指定 Python 版本、安装 uv 包管理器。

```yaml
name: XiaoanNew          # 注意：文件内 name 字段需与实际环境名一致
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - uv
```


### 2.2 pyproject.toml（项目核心配置）

所有业务依赖在 `[project] dependencies` 中声明版本范围（非精确版本）。

当前声明的直接依赖：

| 包 | 版本范围 | 用途 |
|----|---------|------|
| torch | >=2.0.0 | 深度学习框架 |
| torchvision | >=0.15.0 | 视觉工具 |
| ultralytics | >=8.3.0 | YOLOv8 推理 |
| flask | >=2.0.0 | Web API |
| werkzeug | >=2.0.0 | Flask 底层 |
| openai | >=1.0.0 | VLM / OCR 调用 |
| pillow | >=9.0.0 | 图像处理 |
| opencv-python | >=4.8.0 | CV 操作 |
| numpy | >=1.24.0 | 数值计算 |
| pyyaml | >=6.0 | 配置解析 |
| python-dotenv | >=1.0.0 | 环境变量 |
| tqdm | >=4.60.0 | 进度条 |
| tenacity | >=8.0.0 | API 重试 |

开发依赖在 `[project.optional-dependencies] dev` 中：

| 包 | 版本范围 | 用途 |
|----|---------|------|
| pytest | >=7.0.0 | 单元测试 |
| pytest-cov | >=4.0.0 | 覆盖率 |
| ruff | >=0.11.0 | 代码检查与格式化 |
| mypy | >=1.0.0 | 类型检查 |

关键配置段：

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["modules*", "scripts*"]    # 使 modules/ 和 scripts/ 可全局 import

[[tool.uv.index]]
name = "tsinghua"
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true                         # 清华镜像作为默认 PyPI 源

[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"  # PyTorch CUDA 12.1 专用源
```

### 2.3 requirements.txt（版本锁定快照）

由 uv 自动生成，**严禁手动编辑**。当前锁定了 103 个包的精确版本。

生成命令：

```bash
uv pip compile pyproject.toml -o requirements.txt
```


## 3. 环境搭建

### 3.1 新环境部署（全量安装）

```bash
# 第一步：创建 Conda 环境
conda env create -f environment.yml
conda activate XiaoanNew

# 第二步：安装全部依赖（精确版本）
uv pip install -r requirements.txt

# 第三步：可编辑模式安装项目本身（使 modules/ 可全局 import）
uv pip install -e .

# 第四步（可选）：安装开发工具
uv pip install -e ".[dev]"
```

### 3.2 日常激活（已有环境）

```bash
source /root/XiaoanNew/env_setup.sh
```

`env_setup.sh` 做了两件事：
1. `conda activate XiaoanNew`
2. 将环境 bin 目录置于 `PATH` 最前，避免系统 Python 覆盖

---

## 4. 日常开发工作流

### 4.1 新增依赖

使用 `uv add` 直接管理依赖，无需手动编辑 pyproject.toml。

```bash
# 添加运行时依赖
uv add --frozen <包名>
# 示例：uv add --frozen scipy
# 示例（指定版本范围）：uv add --frozen "scipy>=1.10.0"

# 添加开发依赖（仅开发环境需要）
uv add --frozen --dev <包名>
# 示例：uv add --frozen --dev httpx

# 移除依赖
uv remove --frozen <包名>
# 示例：uv remove --frozen scipy
```

`--frozen` 参数的作用：仅修改 pyproject.toml，跳过 uv 自身的 lock/sync 流程（本项目环境由 Conda 管理，不使用 uv 的虚拟环境）。

添加或移除依赖后，需手动刷新锁定文件并同步环境：

```bash
# 重新生成 requirements.txt
uv pip compile pyproject.toml -o requirements.txt

# 同步到当前环境
uv pip install -r requirements.txt
```

完整流程示例（以添加 scipy 为例）：

```bash
uv add --frozen "scipy>=1.10.0"                    # 写入 pyproject.toml
uv pip compile pyproject.toml -o requirements.txt   # 重新锁定
uv pip install -r requirements.txt                  # 安装到环境
```

### 4.2 修改源码

可编辑模式下，对 `modules/` 或 `scripts/` 的改动实时生效，无需重新安装。

验证可编辑安装状态：

```bash
pip show xiaoan-parking-detection | grep "Editable"
# 应输出: Editable project location: /root/XiaoanNew
```

### 4.3 运行测试

```bash
source /root/XiaoanNew/env_setup.sh
pytest                     # 运行全部 347 个测试
pytest --cov=modules       # 带覆盖率报告
```

### 4.4 代码质量检查

```bash
ruff check .               # Lint 检查
ruff format .              # 自动格式化
ruff check . && ruff format --check .   # CI 用：检查不通过则退出码非零
```

---

## 5. 关键机制说明

### 5.1 可编辑安装的工作原理

执行 `pip install -e .` 后，setuptools 在 `site-packages/` 下生成一个 `.egg-link` 文件，
指向 `/root/XiaoanNew/`。Python 的 import 机制会沿此链接找到 `modules/` 和 `scripts/`，
因此在任何目录下都能 `from modules.cv.image_utils import encode_image_to_base64`。

可通过以下命令验证：

```bash
python -c "import modules; print(modules.__file__)"
# 应输出: /root/XiaoanNew/modules/__init__.py
```

### 5.2 PyTorch CUDA 源的解析顺序

`pyproject.toml` 中配置了两个 uv 索引源：

1. **清华镜像**（`default = true`）：所有非 PyTorch 包从此源下载
2. **PyTorch cu121 源**：`torch`、`torchvision`、`torchaudio` 从此源下载 CUDA 版本

uv 在 compile 阶段会自动解析依赖并选择正确的源，生成的 `requirements.txt` 已包含完整的下载 URL。

### 5.3 文件命名禁忌

以下文件名会与 Python 标准库冲突，**严禁使用**：

`test.py`、`math.py`、`io.py`、`os.py`、`sys.py`、`json.py`、`csv.py`、`re.py`、
`abc.py`、`typing.py`、`logging.py`、`random.py`、`collections.py`

---

## 6. 常见问题

**Q: 为什么用 uv 而不是 pip？**

uv 基于 Rust 实现，依赖解析和安装速度远快于 pip。本项目 103 个包的完整安装在 uv 下通常在
30 秒内完成。此外 `uv pip compile` 提供了确定性的版本锁定能力，`pip freeze` 无法做到。

**Q: 为什么 requirements.txt 里没有 `-e .`？**

`requirements.txt` 只管第三方依赖的版本锁定。项目自身的可编辑安装是独立步骤（`uv pip install -e .`），
两者职责分离。混在一起会导致 `uv pip compile` 产生循环引用。

**Q: CUDA 版本不匹配怎么办？**

当前配置：PyTorch 使用 cu121（CUDA 12.1），GPU 驱动支持 CUDA 12.2。驱动版本高于 PyTorch
要求即可向下兼容。如需升级到 cu124，修改 `pyproject.toml` 中 `pytorch-cu121` 索引的 URL
为 `https://download.pytorch.org/whl/cu124`，然后重新 compile。

**Q: 安装速度慢？**

镜像源已在 `pyproject.toml` 的 `[tool.uv.index]` 中配置为清华源，无需额外设置环境变量。
如仍然慢，检查网络是否能访问 `pypi.tuna.tsinghua.edu.cn`。

---

## 7. 目录结构

```
XiaoanNew/
├── pyproject.toml            # 项目核心配置（手动维护）
├── environment.yml           # Conda 环境定义（手动维护）
├── requirements.txt          # 依赖锁定清单（uv 自动生成，禁止手动修改）
├── env_setup.sh              # 环境激活脚本
├── modules/                  # 核心业务逻辑包（可编辑安装，全局可 import）
│   ├── __init__.py
│   ├── config/               #   配置管理
│   ├── cv/                   #   计算机视觉
│   ├── vlm/                  #   大模型客户端
│   ├── experiment/           #   实验管理
│   └── prompt/               #   提示词管理
├── scripts/                  # 实验脚本入口（全局可 import）
│   └── __init__.py
├── tests/                    # pytest 测试套件（347 tests）
├── assets/                   # 配置 / 提示词 / 模型权重
└── xiaoan_parking_detection.egg-info/  # 可编辑安装元数据（自动生成）
```
