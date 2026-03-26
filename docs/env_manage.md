XiaoanNew 纯 uv 环境管理规范 (v2.1.2)
1. 核心设计哲学：全自动闭环
本项目采用 “项目级环境隔离” 策略，利用 uv 实现极速、稳定、可复现的开发体验。
单一入口：pyproject.toml 定义所有逻辑依赖与源路径。
绝对锁定：uv.lock 替代 requirements.txt，确保环境在物理级 100% 一致。
零配置运行：无需手动激活环境，通过 uv run 直接执行。
2. 环境初始化 (新设备/新成员)
第一步：安装 uv
项目不再依赖 Conda。如果电脑没有 uv，请先安装：
Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh
Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
第二步：一键同步环境
进入项目根目录，直接执行以下命令：
code
Bash
# 1. 锁定 Python 版本 (如果系统没有，uv 会自动下载)
uv python pin 3.10

# 2. 一键创建 .venv 并安装所有依赖 (包括 Torch CUDA)
uv sync
注：由于 PyTorch 的 NVIDIA 依赖包较大（约 2GB），首次同步可能需要 2-5 分钟，后续同步将秒级完成。
3. 日常开发工作流
3.1 增加/移除依赖 (自动同步 toml)
严禁手动编辑 pyproject.toml 的 dependencies 字段，请使用 uv 命令自动完成：
需求	命令	效果
增加运行时依赖	uv add pandas	自动下载包 + 写入 toml + 更新 uv.lock
增加开发工具	uv add --dev pytest	自动归类到 [project.optional-dependencies]
移除依赖	uv remove opencv-python	自动清理包并同步配置文件
3.2 运行程序与脚本
无需执行 source .venv/activate。使用 uv run 可以确保程序永远在正确的环境下运行，且自动处理项目路径。
运行主程序：uv run python app.py
运行工具脚本：uv run python scripts/dataset_analysis.py
执行测试：uv run pytest
4. PyTorch CUDA 专项配置规范
为了确保 uv 能够精准命中 NVIDIA 加速版本，pyproject.toml 必须包含以下 “专包专线” 配置：
code
Toml
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://mirrors.bfsu.edu.cn/pypi/web/whl/cu121/" # 推荐使用北外或清华镜像加速
explicit = true  # 强制隔离，防止普通包误入此源

[tool.uv.sources]
# 明确指定这三个核心包走专用 CUDA 通道
torch = { index = "pytorch-cu121" }
torchvision = { index = "pytorch-cu121" }
torchaudio = { index = "pytorch-cu121" }

[tool.uv]
# 设置清华源为默认主源，加速 90% 的普通第三方包
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple"
5. 进阶维护指令
更新项目内所有包到最新版：
uv lock --upgrade
清理不再需要的缓存（通常不需要）：
uv cache clean
查看当前的依赖树结构：
uv tree
导出为传统的 requirements.txt (仅用于向后兼容部署)：
uv export --format requirements-txt -o requirements.txt
6. Git 协作守则
为了保证团队协作不冲突，请务必遵守以下 Git 规则：
必须提交：pyproject.toml 和 uv.lock。
绝对禁止提交：.venv/ (环境目录) 和 *.egg-info/ (中间产物)。
忽略文件示例 (.gitignore)：
code
Text
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
.env