# nwwolf_mjlab

四足组强化学习训练仓库（Black），基于 [MjLab](https://github.com/mujocolab/mjlab) v1.6.0 与 RSL-RL。

本仓库用于队内四足招新培训与队内赛：把一只 12 自由度的四足机器人 **Black**，用 PPO 训练成
能跟着速度指令行走的运动策略。仓库只包含训练侧内容（机器人资产、任务定义、算法、策略导出），
实机部署 runtime 在独立项目中维护。

任务列表：

| Task ID | 内容 |
|---|---|
| `black-flat` | 平地速度跟踪，PPO |
| `black-rough` | 复杂地形（坡面、随机粗糙面、障碍、上下台阶）速度跟踪，PPO |
| `black-flat-him` | 平地 + HIM（history-biased 估计器）算法 |
| `black-rough-him` | 复杂地形 + HIM |

初学阶段只需要 `black-flat` 与 `black-rough`；HIM 属于进阶内容，见第 9 节。

---

## 1. 环境要求

- **操作系统**：Linux（训练依赖 MuJoCo / MJWarp / CUDA）。
- **Python**：3.12（见 `.python-version`）。
- **包管理**：[uv](https://docs.astral.sh/uv/)（不要用 `pip` / `conda` 手工装依赖）。
- **MjLab**：严格使用 **v1.6.0**，已由 `uv.lock` 锁定（RSL-RL 一并锁定）。不要自行升级。
- **GPU**：并行训练必须使用 NVIDIA GPU。默认 `train_num_envs = 4096`，显存需求较高；
  消费级显卡（如 8 GB 显存）请先降低环境数量（见第 4 节）。
  纯 CPU 只能用于 tiny 冒烟测试，不能用来真实训练。

## 2. 安装

```bash
git clone https://github.com/N-W-wolf/nwwolf_mjlab.git
cd nwwolf_mjlab
uv sync --locked
```

`uv sync --locked` 会按 `uv.lock` 精确还原环境（包括 MjLab 1.6.0）。
之后所有命令都用 `uv run` 执行，它会自动使用项目内的虚拟环境。

`train` / `play` 来自 MjLab 自带 CLI，`export` 是本仓库提供的策略导出入口。

## 3. 基础训练（平地）

```bash
uv run train black-flat
```

训练默认写入 `logs/rsl_rl/black_velocity/<时间戳>_flat/`。

默认日志后端是 **wandb**（需要先 `wandb login`）。培训环境通常直接用 TensorBoard：

```bash
uv run train black-flat --agent.logger tensorboard
```

常用覆盖参数（CLI 显式参数优先于任务默认值）：

```bash
--env.scene.num-envs 1024     # 环境数量（默认 4096，显存不足时降低）
--agent.max-iterations 2000   # 训练迭代数；resume 时表示“再训练多少”
--agent.seed 43               # 随机种子
--agent.save-interval 100     # checkpoint 保存间隔
--agent.run-name mytag        # run 目录后缀 → <时间戳>_mytag
```

同 stage 续训（接着上一次 flat 继续）：

```bash
uv run train black-flat --agent.resume True
```

resume 时不写 `--agent.load-run`，会自动选择最新的 `*_flat` run 与其中最新的 `model_*.pt`。

## 4. 复杂地形训练（rough）

`rough` 与 `flat` 属于同一个 checkpoint-compatible policy family：网络结构、PPO 超参数一致，
因此可以从已经训练好的 flat 策略继续训练复杂地形（推荐做法，比从零训练快且稳）。

从最新的 flat checkpoint 续训 rough：

```bash
uv run train black-rough \
    --agent.resume True \
    --agent.load-run '.*_flat$' \
    --agent.logger tensorboard
```

指定某个 flat run 与具体 checkpoint：

```bash
uv run train black-rough \
    --agent.resume True \
    --agent.load-run '<flat-run>' \
    --agent.load-checkpoint 'model_500.pt'
```

说明：

- `--agent.load-run` 接受 run 目录名或正则；`.*_flat$` 表示“最新的 flat run”。
- flat → rough 是 **full resume**：actor / critic / optimizer / normalization / iteration 等训练状态
  全部继承（包括 checkpoint 中的学习率）；rough 的环境、地形与 curriculum 会重新构造。
- **resume 时 `--agent.max-iterations` 表示“在 checkpoint 之后再训练多少 iteration”**，不是最终编号。
  例如 checkpoint 在 iter 1000，`--agent.max-iterations 4000` 会再训练约 4000 iterations。
- 直接从零训练 rough 也可以（`uv run train black-rough`），但需要更长时间。

## 5. 策略回放

用训练产生的 checkpoint 回放，必须显式给出 checkpoint 文件：

```bash
uv run play black-flat \
    --checkpoint-file 'logs/rsl_rl/black_velocity/<flat-run>/model_500.pt'
```

`<flat-run>` 需要替换成实际的 run 目录名。

常用参数：

```bash
--viewer viser     # 浏览器网页界面（默认 http://localhost:8080，可拖动速度指令滑条）
--viewer native    # MuJoCo 桌面窗口（需要 DISPLAY）
--viewer auto      # 默认：检测到桌面环境用 native，否则用 viser
--num-envs 4       # 同时显示多个机器人
--agent zero|random|trained   # policy 模式，zero/random 用于检查环境本身
--device cpu
```

`--viewer auto` 在桌面会话下会落到 native 窗口，此时**没有**速度指令滑条。想手动改变速度指令，
请显式使用 `--viewer viser`。

回放 rough 会把地形一起生成出来；回放只加载 actor，不使用 critic。

## 6. 训练结果

```text
logs/rsl_rl/black_velocity/
├── 2026-01-01_12-00-00_flat/
│   ├── model_0.pt
│   ├── model_500.pt
│   ├── model_1000.pt
│   └── ...
└── 2026-01-01_13-00-00_rough/
```

- `model_<iteration>.pt`：训练 checkpoint（actor + critic + optimizer + 训练状态），
  用于 resume / play / export。
- `save_interval` 决定定期保存间隔，最后一个 iteration 一定保存。
- 不同环境数量训练出来的 run 可以互相 resume，checkpoint 格式不受 env 数量影响。

用 TensorBoard 查看训练曲线（仅在 `--agent.logger tensorboard` 时产生事件文件）：

```bash
tensorboard --logdir logs/rsl_rl
```

## 7. 修改训练参数

| 文件 | 职责 | 可以改什么 |
|---|---|---|
| `src/alldog_mjlab/tasks/velocity/black/black_config.py` | 日常训练参数的**唯一入口** | reward 权重、command 范围、noise、reset、termination、DR、terrain、PPO 超参、迭代数、环境数量 |
| `src/alldog_mjlab/tasks/velocity/black/rewards.py` | reward 项的具体实现 | 新增 / 修改单个 reward 函数 |
| `src/alldog_mjlab/tasks/velocity/black/env_cfgs.py` | 把上面的数值装配成 MjLab 环境 | 新增 reward / observation / event / termination 后在此注册 |
| `src/alldog_mjlab/robots/black/black_constants.py` | 机器人固有参数 | actuator（PD）、default pose、effort limit 等 |

调参的基本顺序建议：**先在 `black_config.py` 改数值 → 起短跑看 reward 与 `Episode/rew_*` 曲线。**
只有需要新的行为（新的 reward 项、新的观测项）时才去动 `rewards.py` / `env_cfgs.py`。

⚠️ `black_config.py` 中有几项属于 **policy / 部署契约**，不是普通超参数，改动会导致已训练策略
和部署端不兼容：`policy_dt`（= `physics_dt × decimation`）、`action_scale`、actor observation scales。
joint order / observation layout / term 顺序定义在 `env_cfgs.py` 与 `robots/black/`，同样属于契约。
修改前请先和队内负责部署的人确认。

## 8. Black 关键契约（不要随手改）

```text
policy joint order : FL → FR → RL → RR，每腿 hip → thigh → calf
actor observation  : 单帧 45 维（PPO，无 history、无 running normalization）
action             : 12 维，q_target = q_default + 0.25 × raw_action
physics dt         : 0.005 s
decimation         : 4      → policy dt = 0.02 s（50 Hz）
nominal PD         : Kp = 40，Kd = 1.2
```

observation 依次为：command(3) → base angular velocity(3) → projected gravity(3) →
joint position 相对 default(12) → joint velocity(12) → previous action(12)。

注意 **MuJoCo 模型里的腿顺序（FL → FR → RR → RL）与 policy 顺序不同**，导出 / 部署时必须显式映射。

## 9. 进阶内容（可选）

### 课程学习（command curriculum）

`black_config.py` 中的 forward-speed command curriculum 默认开启：策略跟踪表现达标后，
自动把前进速度指令范围从 `±1.0 m/s` 逐步扩展到 `±2.0 m/s`（其余指令范围固定）。
resume 时 curriculum 状态的恢复策略由 `--agent.command-curriculum-restore` 控制
（`auto` / `full` / `range` / `none`），默认 `auto`：同 stage 精确续训，跨 stage 只继承速度范围。

### HIM

`black-flat-him` / `black-rough-him` 在相同环境契约上叠加 HIM 算法：actor 输入为
6 帧 × 45 维 = 270 维的 history，额外训练一个从 history 估计速度与隐变量的估计器
（target encoder），PPO 侧超参数与普通 PPO 同源。

```bash
# 1) 平地 HIM：从已训练好的 black-flat PPO checkpoint 初始化（warm start，非 resume）
uv run train black-flat-him --agent.warm-start True --agent.load-run '.*_flat$'

# 2) 地形 HIM：从 flat HIM checkpoint 完整续训（rough HIM 不支持 PPO→HIM warm start）
uv run train black-rough-him --agent.resume True --agent.load-run '.*_flat_him$'

# 回放方式与普通 PPO 一致
uv run play black-flat-him --checkpoint-file 'logs/rsl_rl/black_velocity/<flat-him-run>/model_500.pt'
```

`--agent.warm-start` 与 `--agent.resume` 互斥：前者是“初始化一个新 run”，后者是“恢复训练状态”。

### 策略导出（TorchScript）

```bash
uv run export --task-id black-flat
uv run export --task-id black-rough --load-run '<run>' --checkpoint model_500.pt --output-dir ~/models/black
```

默认流程为 `task → 最新同 stage run → 最新 checkpoint → <run>/exported/policy.pt`，在 CPU 上导出。

| 产物契约 | 值 |
|---|---|
| 格式 | TorchScript，仅 actor |
| 输入 | `float32 [1, 45]`（PPO）/ `float32 [1, 270]`（HIM） |
| 输出 | `float32 [1, 12]`，raw policy action |
| 文件名 | `policy.pt` |

训练 checkpoint `model_*.pt` 与部署用的 `policy.pt` 用途不同，不能互换。

## 10. 关于机器人模型

培训版使用**简化视觉模型**：仓库不包含任何 STL / mesh 文件（`black.xml` 中已移除 mesh 资产），
机器人外观由 MuJoCo 原生几何体（box / cylinder / sphere）构成，便于发布和快速加载。

以下内容与完整模型**完全一致**，视觉简化不影响训练结果：

- 全部 body 层级与 19 个显式 inertial（质量 / 质心 / 惯量矩阵）
- 全部 joint 及其 limit、joint 顺序
- 所有碰撞几何体的位置、尺寸、摩擦与接触参数
- contact exclusion、default、site / IMU / sensor 结构
- actuator、default pose、effort limit 等控制参数

因此本仓库训练出的策略与完整模型训练出的策略行为一致，可以正常导出并部署。

## 11. 目录结构

```text
src/alldog_mjlab/
├── robots/black/          # Black 资产（MJCF）、joint/actuator 身份、固有参数、default pose
├── tasks/velocity/black/  # 观测、动作、command、reward、termination、reset/event、terrain、DR
├── algorithms/him/        # HIM estimator / HIMPPO / warm start
└── utils/export_policy.py # TorchScript 策略导出 CLI
```

## 12. 常见问题

| 现象 | 处理 |
|---|---|
| 训练一开始就要求 wandb 登录 | 加 `--agent.logger tensorboard`，或先 `wandb login` |
| CUDA out of memory | 降低 `--env.scene.num-envs`（如 1024 / 512） |
| `play` 弹不出窗口 | 无桌面会话时用 `--viewer viser`，用浏览器打开显示的地址 |
| rough play 想调速度指令 | 显式加 `--viewer viser` |
| 不确定当前环境是否是锁定版本 | 重新执行 `uv sync --locked`，不要手工 `pip install` |
