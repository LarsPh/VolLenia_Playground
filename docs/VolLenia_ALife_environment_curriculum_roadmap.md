# VolLenia 论文路线图：环境课程、生命体规则复杂度与可微搜索工程设计

> 版本：2026-06-19  
> 目标：把近期关于 VolLenia、Lenia / NCA / ALife、Sensorimotor Lenia、环境与 agent 设计、可微搜索、3D procedural 环境的讨论整理成一份可执行的研究路线图。

---

## 0. 总结：我们真正要做什么

目前比较清晰的目标可以概括为：

> **通过可微搜索、自动课程和环境压力，在 3D 连续 CA / Lenia-like 体场中发现能自维护、导航、避障、竞争资源、与其他个体互动的 volumetric artificial life agents。**

这不是单纯做一个 3D Lenia renderer，也不是把 NCA 的图像复原 demo 换成 3D。更像是把三条路线合并：

1. **Lenia / Flow Lenia 的连续生命动力学**：连续密度场、局部卷积相互作用、软体生命感、质量守恒、参数局部化、多物种互动。
2. **NCA / differentiable CA 的可训练性**：把局部规则、感受野、growth function 或 hidden channels 放进可微图，用 BPTT 和目标函数搜索行为。
3. **ALife / RL 环境设计的 curriculum 思想**：不是手动挑几个漂亮 preset，而是构造一系列环境压力，让生命体能力随环境复杂度逐步出现。

最终 VolLenia 可以被定位成：

```text
VolLenia = 3D continuous artificial-life substrate
         + differentiable/searchable local rules
         + obstacle/resource/toxicity/flow/environment channels
         + curriculum and archive-based discovery
         + real-time volumetric visualization
```

论文叙事上，真正重要的不是“我们做了 3D Lenia”，而是：

> **我们提出一个可微分、可搜索、可交互可视化的 3D continuous CA 世界，并展示如何通过环境课程发现具有类生命 sensorimotor behavior 的 volumetric agents。**

---

## 1. 三条主线

这份路线图分成三条互相耦合的主线：

### 主线 A：环境复杂度路线图

从最简单的“活着”开始，逐步加入障碍、门、资源、毒性、动态资源、流场、多 agent、程序化 3D 几何。

环境的目标不是装饰，而是制造生命压力：不行动会死，不适应会被破坏，不探索会耗尽资源，过度扩张会中毒或崩溃。

### 主线 B：生命体能力 / 规则复杂度路线图

生命体规则不应一开始就 full neural / full ecology。更稳妥的是随环境复杂度逐步增加规则表达能力：

```text
classic Lenia parameters
→ multi-channel VolLenia
→ Flow / mass-conservative VolLenia
→ resource metabolism
→ chemical / olfactory sensing
→ anisotropic / orientation-aware perception
→ neural growth / learned local physics
→ local genotype / multi-species interaction
```

### 主线 C：搜索方法与计算效率路线图

方法上用 **outer archive / diversity / curriculum search + inner BPTT gradient descent**。工程上用 **PyTorch 可微研究后端 + C++/CUDA 实时展示后端**。先做低分辨率可微搜索，再把参数导入高分辨率 C++ VolLenia 里展示。

---

# Part I：环境复杂度路线图

## 2. 环境设计的核心原则

### 2.1 环境必须提供 survival pressure

如果 agent 不行动、不适应、不维持自身也能一直存在，就很难诱发复杂行为。环境应当让生命体处在一种“precarious” 状态：它必须持续维持自己，否则死亡、扩散、爆炸、被腐蚀或耗尽资源。

可用机制：

```text
自然衰减：density slowly decays
资源依赖：growth requires local resource
障碍侵蚀：obstacle field causes negative growth
毒性积累：waste inhibits growth
运动成本：flow / metabolism consumes mass or resource
局部竞争：多个 agent 争同一资源
```

### 2.2 环境应通过局部场影响 agent，而不是直接给全局答案

更 ALife 的设计不是给 agent 一个 `target_vector`，而是让环境通过局部物理场影响它：

```text
resource diffusion field
obstacle SDF / repulsive potential
toxicity / waste field
flow field
temperature / light field
chemical attractor / repellent
```

这样 agent 的感知与身体耦合。它不是传统 RL agent 那样“看见整张地图再输出动作”，而是通过自身结构在局部场中的变形、增长、收缩、流动来产生 sensorimotor behavior。

### 2.3 环境难度要处于“刚好能学”的边界

如果环境太简单，系统学不到复杂行为；如果太难，所有候选都会死掉。需要 curriculum：

```text
如果成功率过高 → 增加障碍密度、目标距离、毒性、资源稀缺度
如果成功率过低 → 降低障碍、增加资源、缩短目标距离、减少动态扰动
```

这和 POET、PAIRED、UED 等自动课程思想一致：好的环境不是固定的，而是跟随 agent 当前能力调整。

### 2.4 环境要提供 stepping stones

复杂行为通常不是直接出现，而是通过中间能力逐步堆叠：

```text
localized body
→ stable survival
→ self-propulsion
→ directional movement
→ obstacle recovery
→ obstacle avoidance
→ resource seeking
→ resource exploration
→ multi-agent competition
→ niche formation / symbiosis / reproduction
```

因此 archive 不应只存“最好”的个体，而应存多样的能力类型。某个慢但稳定的个体，可能是后续避障个体的 stepping stone；某个脆但快的个体，可能是后续 resource-seeking 的 stepping stone。

### 2.5 环境要有可复用 grammar

如果完全手写环境，会很快枯竭；如果完全随机生成，很多环境无意义。更好的方式是设计环境 grammar：

```text
primitive geometry:
    sphere obstacle, wall, tunnel, arch, gap, chamber, slope, moving block

field primitives:
    resource blob, toxic plume, attractor, repellent, flow vortex

composition rules:
    resource behind wall
    tunnel connects chambers
    moving hazard crosses corridor
    toxic region surrounds resource
    resource patches deplete sequentially
```

这允许自动生成大量有结构、可调难度、可解释的环境。

---

## 3. 环境课程 Level 0–9

下面是一个建议课程。每一级都包含环境设计、能力目标、可量化指标、典型失败模式。

---

## Level 0：保持不死不爆，形成局部体

### 环境

```text
无障碍
无资源或均匀资源
无目标点
无其他 agent
```

### 目标

最基础的生命体存在性：

```text
不死亡
不全场爆炸
保持局部化
质量稳定
形态有连贯结构
```

### 指标

```text
mass_min < total_mass < mass_max
compactness / second moment
connected component count
bounding radius
entropy / spread
long-horizon stability
```

### 失败模式

```text
全灭：density → 0
爆炸：density 填满整个 volume
雾化：质量扩散成大范围云
碎裂：分成很多无关片段
静态死物：稳定但没有后续行为潜力
```

### 备注

这是所有后续任务的基础。如果没有 reliable alive filter，后面导航和避障 loss 很容易被“全场填满目标区域”这种作弊解利用。

---

## Level 1：自发移动 / 方向性

### 环境

```text
无障碍
可选弱目标点
可选弱资源梯度
```

### 目标

让局部体产生持续位移，而不是仅仅静止存在：

```text
center of mass 有持续移动
移动过程中质量稳定
形态保持连贯
速度不要剧烈抖动
```

### 指标

```text
COM displacement
mean velocity
velocity direction consistency
mass preservation during movement
compactness during movement
```

### 失败模式

```text
漂移太慢
靠喷射质量移动，主体死亡
主体被拉成长条或碎裂
全场扩散导致 COM 移动是假象
```

### 课程设计

可以从短 horizon 开始：

```text
T = 16 steps → T = 32 → T = 64
目标距离：近 → 中 → 远
```

---

## Level 2：单一资源趋化

### 环境

```text
一个 resource blob
resource field 可扩散
agent growth 依赖 resource
resource 被 agent 消耗
```

### 目标

让 agent 从“朝坐标移动”转向“追求生命资源”：

```text
靠近 resource
吸收 resource
在 resource 周围维持或增长
resource 消耗后不爆炸
```

### 指标

```text
distance(COM, resource_center)
resource intake
mass / energy balance
resource overlap
survival time after resource depletion
```

### 失败模式

```text
只在初始位置活着，不追资源
资源导致爆炸增长
追到资源后失控扩散
通过伸出细丝碰资源作弊
```

### 设计要点

资源应该既是吸引因素，又是约束因素。可以设计：

```text
growth = growth_lenia + α * resource_local - β * waste
resource_{t+1} = resource_t - consumption_rate * density + diffusion + regeneration
```

---

## Level 3：接触障碍 / 稀疏障碍

### 环境

```text
随机 obstacle spheres
obstacle channel 固定
obstacle 对 agent 产生局部负增长或不可占据约束
```

### 目标

agent 能够在遇到障碍后：

```text
减少碰撞
碰撞后恢复结构
绕过稀疏障碍
继续到达资源或目标
```

### 指标

```text
obstacle overlap
post-collision mass recovery
success rate over random obstacle seeds
trajectory deviation
recovery time
```

### 失败模式

```text
一碰即死
被障碍切成多块后无法恢复
直接穿透障碍
学会远离所有东西但也不接近资源
```

### 备注

Sensorimotor Lenia 的障碍物基本就是这个模式：obstacle channel 对 learnable channel 造成局部破坏，agent 通过身体被扰动来“感知”障碍，类似触觉。

---

## Level 4：墙、门洞、窄通道、房间

### 环境

```text
wall plane
wall with gap
tunnel
corridor
room-to-room layout
arch / cave opening
```

### 目标

从“躲避散点障碍”升级为“利用空间结构导航”：

```text
通过 gap
沿 corridor 前进
从 chamber A 到 chamber B
在窄通道中保持身体完整
```

### 指标

```text
passage success rate
minimum clearance handled
time-to-resource
body deformation / compression ratio
recovery after tunnel
```

### 失败模式

```text
卡在墙前
在窄通道中碎裂
只会绕开，找不到门洞
挤压后无法恢复形态
```

### 环境来源

这一阶段可以借鉴：

```text
MiniGrid / MiniWorld 的房间、门、钥匙、走廊 grammar
ProcTHOR / AI2-THOR 的室内房间布局
Infinigen Indoors 的 constraint-based arrangement 思路
```

对 VolLenia 来说，只需要几何结构，不需要真实材质和视觉渲染。

---

## Level 5：资源耗尽与探索

### 环境

```text
多个 resource patches
单个资源会被消耗
资源缓慢再生或不再生
可选 toxic waste 随消耗产生
```

### 目标

agent 不能只停在一个地方，必须探索新资源：

```text
发现资源 patch
消耗后离开
探索下一处资源
在资源稀缺下维持生命
```

### 指标

```text
number of patches visited
resource intake over time
survival time
trajectory coverage
response latency after depletion
```

### 失败模式

```text
只吃初始资源后死亡
无限停留在耗尽区域
为了探索而碎裂扩散
过度增长导致资源快速耗尽
```

### 设计要点

这个阶段开始像生态系统。推荐加入 waste：

```text
resource → consumed by density
waste → produced by density
waste inhibits growth
waste diffuses or decays
```

这样会自然产生“不能一直待在原地”的压力。

---

## Level 6：毒性、禁区、风险-收益权衡

### 环境

```text
resource 被 toxic field 包围
toxic plume 随时间扩散
某些区域 resource 高但毒性高
```

### 目标

让 agent 学会在资源收益和死亡风险之间折中：

```text
短暂进入高资源区
避免长时间停留毒区
绕开高毒障碍
利用低毒路径
```

### 指标

```text
resource intake / toxicity exposure ratio
toxic overlap
survival after toxic encounter
path efficiency
```

### 失败模式

```text
贪资源进入毒区死亡
完全避开毒区导致资源不足
用扩散身体同时吃所有资源，失去个体性
```

### 备注

这个阶段可以借鉴 RL 中的 lava / hazard / risk-reward 任务，但实现为连续场：

```text
toxicity(x) 对 growth 产生负项
resource(x) 对 growth 产生正项
```

---

## Level 7：动态资源、动态障碍、流场

### 环境

```text
moving obstacles
opening / closing gates
resource pulses
flow field / vortex / current
toxic waves
```

### 目标

适应时间变化：

```text
避开移动 hazard
等待门打开
跟随资源脉冲
抵抗流场拉扯
利用流场移动
```

### 指标

```text
success rate under dynamic seeds
phase-locking behavior
recovery after perturbation
time-varying collision rate
energy cost under flow
```

### 失败模式

```text
只能处理静态环境
被流场撕裂
遇到动态障碍直接死亡
无法等待，只会盲目前进
```

### 设计来源

可借鉴：

```text
OpenAI Hide-and-Seek 中 moving/usable object 的思路
Infinigen-Sim 的 articulated / dynamic object idea
机器人导航中的 moving obstacle / dynamic maze
```

早期不需要真实刚体物理。可以先用简单时变体素场实现：

```text
O(x,t) = moving_sphere(t)
R(x,t) = pulsing_resource(t)
F(x,t) = rotating_flow_field(t)
```

---

## Level 8：多 agent 资源竞争

### 环境

```text
两个或多个 agent
共享 resource patches
agent 之间可互相排斥、腐蚀、遮挡或竞争资源
```

### 目标

引入真正的 interaction：

```text
竞争同一资源
避让其他 agent
追逐资源
形成空间分区
可能出现共存
```

### 指标

```text
relative resource intake
agent-agent distance
coexistence time
collision / overlap
niche separation
population stability
```

### 失败模式

```text
一个 agent 总是立即消灭另一个
所有 agent 融合成一团
资源竞争不产生行为差异
系统很快崩溃或爆炸
```

### 设计参考

Melting Pot / Social Dilemma 类环境的核心不是物理，而是资源分配与社会压力：

```text
公共资源
过度采集导致生态崩溃
个体短期收益与群体长期收益冲突
```

在 VolLenia 中可以变成：

```text
shared resource field
regeneration rate depends on local density
waste produced by over-consumption
```

---

## Level 9：捕食、共生、繁殖、生态循环

### 环境

```text
多种 genotype / species
一种 agent 的 waste 是另一种 agent 的 resource
死亡后变成 resource
可分裂 / budding / reproduction
资源链条和转化链条
```

### 目标

从单个 embodied agent 走向 ecology：

```text
捕食
寄生
共生
分裂繁殖
长期 population dynamics
niche formation
```

### 指标

```text
population count
birth / death rate
species diversity
coexistence duration
resource cycle stability
long-run non-collapse
```

### 失败模式

```text
一物种迅速垄断全场
系统周期性全灭
繁殖只是噪声碎裂
生态循环不稳定
```

### 备注

这一级非常难，不应作为第一篇 paper 的主目标。但它是 VolLenia 的长期方向。

---

# Part II：3D Procedural 环境与几何转换路线

## 4. 为什么需要现成复杂环境库

完全手写环境会很慢；完全随机 primitive 又很容易生成无意义结构。因此需要借鉴现有复杂环境库的 **geometry grammar** 和 **task grammar**。

我们的目标不是直接用这些库训练 Lenia，而是抽取它们的可用结构：

```text
几何结构：墙、洞、坡、洞穴、走廊、房间、障碍、自然地形
任务结构：目标、资源、危险、门、钥匙、通路、动态障碍、多 agent 竞争
课程结构：从简单到复杂、自动调难度、成对进化、regret-based challenge
```

---

## 5. Infinigen 作为几何生成参考

Infinigen 的价值在于：它能生成极其多样的自然场景几何。我们不一定需要它的 photorealistic material 和 rendering，而是需要：

```text
terrain heightfield
rocks
caves
arches
plants / tree-like obstacles
water / cloud / volumetric structures
natural clutter
```

对 VolLenia，可离线转换为：

```text
O(x,y,z): obstacle occupancy
SDF(x,y,z): signed distance field
R(x,y,z): resource field
T(x,y,z): toxicity / hazard field
F(x,y,z): flow field
M(x,y,z): semantic material label
```

### 5.1 推荐转换流程

```text
1. 生成或导入 mesh / terrain
2. 统一 scale 到 Lenia volume world
3. voxelization → occupancy grid O
4. compute SDF → distance-to-obstacle field
5. 根据 semantic label 或 heuristic 放置 resource/toxicity
6. downsample 到训练分辨率 32^3 / 64^3
7. 保存为 npz / torch tensor / volume cache
8. C++ 展示端使用高分辨率版本
```

### 5.2 Infinigen 的使用边界

Infinigen 很适合离线生成复杂环境库，但不适合作为训练 inner loop 的实时环境生成器。训练时应使用预生成 cache 或简化 grammar。

---

## 6. Infinigen Indoors / ProcTHOR / Habitat 的启发

### Infinigen Indoors

有价值的不是材质，而是 constraint-based indoor layout：

```text
房间
走廊
门洞
家具障碍
半封闭空间
遮挡
多房间连接
```

这些可以转为 3D Lenia 的 tunnel / chamber / wall / gap 课程。

### ProcTHOR / AI2-THOR

适合借鉴 embodied AI 的导航任务：

```text
从房间 A 到房间 B
绕过家具障碍
通过门洞
目标物在另一个房间
对象作为资源或毒性源
```

### Habitat / BEHAVIOR / OmniGibson

适合借鉴 long-horizon task grammar：

```text
寻找资源
穿越空间
移动物体
打开通道
重排对象
维持环境状态
多阶段任务链
```

VolLenia 不需要真实物体交互，但可以把这些任务抽象成连续体场问题。

---

## 7. 更轻量的几何生成替代方案

如果 Infinigen 太重，可以先做自定义 geometry grammar：

```text
Primitive set:
    sphere
    capsule
    box
    plane
    torus
    cylinder
    heightfield
    tunnel
    arch
    chamber
    wall_with_gap

CSG operations:
    union
    subtraction
    intersection
    smooth union

Field outputs:
    occupancy
    SDF
    normal / gradient
    material label
```

这对可微搜索更友好，也方便自动调难度。

### 推荐最小实现

```python
EnvironmentSpec:
    primitives: list[Primitive]
    resources: list[ResourcePatch]
    toxic_fields: list[ToxicPatch]
    flow_fields: list[FlowPrimitive]
    dynamic_events: list[Event]
```

然后编译为：

```text
O: obstacle occupancy
D: obstacle SDF
R: resource
T: toxicity
F: flow vector field
```

---

# Part III：环境 curriculum 策略

## 8. 手工 staircase curriculum

最简单、最可控：

```text
if success_rate > threshold_high:
    increase difficulty
if success_rate < threshold_low:
    decrease difficulty
```

可调参数：

```text
target distance
obstacle count
obstacle radius
gap width
resource amount
resource regeneration rate
toxicity strength
flow magnitude
dynamic obstacle speed
number of agents
```

优点：容易实现、debug 方便。缺点：人工设计多，可能错过 stepping stones。

---

## 9. Archive-based goal exploration / IMGEP

这是 Sensorimotor Lenia 最值得直接继承的框架。

```text
archive stores:
    θ: agent/rule parameters
    E: environment parameters
    b: reached behavior descriptor
    score: survival / success / robustness

loop:
    sample target behavior g*
    find archived b closest to g*
    copy corresponding θ
    mutate θ or E
    inner gradient descent with BPTT
    evaluate over environment seeds
    add new θ, E, b, score to archive
```

### behavior descriptor 可以包括

```text
final COM position
velocity
mass
compactness
resource intake
obstacle overlap
survival time
number of connected components
trajectory descriptor
multi-agent distance / resource share
```

### 为什么适合 Lenia

因为 Lenia 参数空间中随机点大多死亡、爆炸或静止。archive 允许从“已经活着”的候选开始局部优化，显著提升成功率。

---

## 10. POET-like 成对进化

POET 的核心是：环境和解法成对共同进化，并允许解法跨环境迁移。

VolLenia 版：

```text
archive entry = (environment E_i, agent θ_i, score_i)

loop:
    mutate environment E_i → E_j
    select source θ from same pair or another pair
    optimize θ on E_j
    if E_j neither too easy nor too hard:
        add (E_j, θ_j) to archive
```

### 环境突变方式

```text
increase obstacle density
narrow gap
move resource behind wall
add toxic region
add flow perturbation
add moving hazard
split resource into multiple patches
add competing agent
```

### 接受环境的条件

```text
not too easy:
    current archive best success < upper_threshold

not impossible:
    at least one archived agent can survive / partially solve
```

### 优点

帮助系统自动发现 stepping stones，而不是只沿着人工定义的一条路线前进。

---

## 11. Regret-based / PAIRED-like 环境生成

PAIRED 的思想可以简化成：好的环境是“某个强 agent 会，但当前 agent 不会”的环境。

VolLenia 版：

```text
current agent θ_current
archive best agents {θ_best}

for candidate environment E:
    score_best = max score(θ_best, E)
    score_current = score(θ_current, E)
    regret = score_best - score_current

choose E with high regret but nonzero solvability
```

这可以避免两个问题：

```text
纯随机环境：大量无意义或太难
纯对抗环境：容易生成无解环境
```

### 可用 score

```text
survival time
resource intake
target reached
obstacle collision rate
alive + compact + reached composite score
```

---

## 12. 对抗目标与多 agent autocurriculum

多 agent 对抗是 complexity amplifier。

最小版本：

```text
两个 agent 争同一资源
agent A 优化 resource intake
agent B 也优化 resource intake
resource 有限
```

复杂一点：

```text
prey: survive and eat resource
predator: maximize overlap / consume prey
prey evolves avoidance
predator evolves pursuit
```

或者：

```text
producer: 把 resource 转成 metabolite
consumer: 依赖 metabolite 生长
```

### 注意

不要太早进入复杂多 agent。最好等单 agent 已经有稳定 locomotion、resource seeking、obstacle recovery 后再加入。否则搜索空间会爆炸。

---

## 13. LLM 辅助课程设计

LLM 不应直接替代低层搜索，但可以作为 high-level curriculum planner。

可用方式：

### 13.1 Bottleneck diagnosis

给 LLM 输入统计摘要：

```text
success rate
failure categories
trajectory descriptors
collision rate
resource intake
mass collapse patterns
```

让它提出下一批环境修改建议：

```text
降低障碍密度
增加资源梯度
把目标放近一点
改成更宽通道
增加局部 resource cue
```

### 13.2 Environment grammar proposal

让 LLM 在已有 primitives 上组合新任务：

```text
resource behind two staggered walls
narrow tunnel followed by safe chamber
moving toxic plume crossing path
three resource patches with depletion order
```

### 13.3 Behavior labeling

LLM 可用于辅助人类给视频/指标打标签：

```text
像绕行
像追逐
像分裂
像休眠
像共生
```

但论文主指标仍要用数值指标，LLM label 只能作为探索工具或辅助分析。

---

# Part IV：生命体能力与规则复杂度路线图

## 14. 规则复杂度不要一开始拉满

一个推荐原则：

> **环境复杂度每提升一级，规则复杂度只提升一点。**

如果一开始同时加入 3D、multi-channel、Flow、resource、toxicity、anisotropic kernels、MLP、multi-agent，系统会完全不可 debug。更稳路线是：先让经典参数化 Lenia 在简单环境中工作，再逐步神经化和生态化。

---

## 15. Rule Level A：经典参数化 VolLenia

### 状态

```text
A: density
```

### 更新

```text
U = Kθ * A
G = growthθ(U)
A_next = clip(A + dt * G)
```

### 可训练参数

```text
kernel radius
kernel bands
growth μ / σ
time scale
initial patch
```

### 适用任务

```text
Level 0: survival/localization
Level 1: movement
```

### 优点

简单、可解释、容易和现有 VolLenia C++ 实现对齐。

### 局限

没有资源、环境、隐藏记忆，复杂交互能力有限。

---

## 16. Rule Level B：multi-channel VolLenia

### 状态

```text
A: creature density
H1...Hk: hidden/internal channels
O: obstacle channel
R: resource channel
T: toxicity/waste channel
```

### 更新

```text
A_next = f_A(A, H, O, R, T)
H_next = f_H(A, H, O, R, T)
R_next = resource_dynamics(R, A)
T_next = toxicity_dynamics(T, A)
```

### 适用任务

```text
obstacle
resource seeking
toxicity
recovery
```

### 意义

hidden channels 可以承担类似内部状态、相位、方向、记忆、局部化学信号的角色。这是从 Lenia 向 NCA 靠近的第一步。

---

## 17. Rule Level C：Flow VolLenia / 质量守恒

### 动机

普通 Lenia 的增长/死亡可能导致质量凭空出现或消失，生态和多物种长期互动不稳定。Flow Lenia 的思想是引入质量守恒和参数局部化，让生命体更像物质流而不是纯密度生成。

### 形式

```text
A_next = advect(A, velocity_field) + local_growth_or_exchange
velocity_field = function(local_potentials, gradients, hidden_state)
```

或者：

```text
mass moves rather than appears/disappears
resource converts into biomass
biomass converts into waste / dead matter
```

### 适用任务

```text
long-horizon survival
multi-agent coexistence
resource competition
ecology
```

### 风险

实现和可微搜索难度提高。建议在 Level 0–3 跑通后再加入。

---

## 18. Rule Level D：代谢与资源耦合

### 状态

```text
A: biomass
R: resource
W: waste
E: internal energy optional
```

### 更新示例

```text
resource_consumption = c * A * R
A_growth = growth_lenia + α * resource_consumption - β * toxicity
R_next = R - resource_consumption + diffusion(R) + regeneration
W_next = W + waste_rate * A - decay(W) + diffusion(W)
```

### 能诱发的行为

```text
趋化
觅食
资源耗尽后迁移
避免 waste
竞争资源
休眠 / 复苏
```

这是从“形态生命”转向“生态生命”的关键。

---

## 19. Rule Level E：嗅觉 / 化学感受野

### 为什么先做嗅觉而不是视觉

嗅觉/化学感受更符合局部场模型：资源或毒性通过 diffusion 形成梯度，agent 只需要感受局部浓度和梯度。

### 感知量

```text
local_resource = K_R * R
resource_gradient = ∇(K_R * R)
local_toxicity = K_T * T
toxicity_gradient = ∇(K_T * T)
```

### 可学习部分

```text
growth = f(local_density, resource_potential, toxicity_potential, gradients)
```

### 适用任务

```text
resource seeking
toxic avoidance
multi-resource exploration
trail following
```

### 优点

保持局部性、可微、易于实现、和 ALife 气质一致。

---

## 20. Rule Level F：各向异性 / 朝向感受野

### 动机

如果 agent 想更像生物，只有接触感受和各向同性化学感受可能不够。视觉、前后方向、触须、定向探测都需要各向异性感受野。

### 基础形式

原始 Lenia kernel 通常是 isotropic ring：

```text
K(d) = K(|d|)
```

各向异性 kernel 可以写成：

```text
K(d, u) = radial(|d|) * angular(dot(normalize(d), u))
```

其中 `u` 是方向。

### 方向来源

#### 方案 1：固定方向 basis

```text
K_0, K_1, ..., K_n 对应不同方向
agent 用 hidden orientation 权重混合这些 kernel
```

优点：实现简单。缺点：方向分辨率有限。

#### 方案 2：局部 orientation field

```text
state = [density, hidden, orientation_x, orientation_y, orientation_z]
```

每个 cell 自己维护方向。感受野随局部方向旋转。

优点：最分布式、最像生物。缺点：实现复杂、训练不稳定。

#### 方案 3：body-level orientation

用整个 agent 的 COM velocity 估计方向，然后使用统一视锥。

优点：简单有效。缺点：引入全局计算，CA 味道弱。

### 视觉锥形感受野

可以设计：

```text
vision_cone(d, u) = exp(-|d| / scale) * sigmoid(k * (dot(d_hat, u) - cos(angle)))
```

如果要做遮挡，还需要 ray / visibility 或基于 SDF 的 soft occlusion。这会显著提高复杂度，建议后期再做。

### 推荐顺序

```text
contact obstacle
→ resource diffusion / chemotaxis
→ fixed directional kernels
→ learned anisotropic kernels
→ local orientation field
→ soft vision cone with occlusion
```

---

## 21. Rule Level G：可学习 growth / neural local physics

### 中间路线：Lenia perception + small MLP update

不是一上来 full NCA，而是保留 Lenia 的卷积感知，用小 MLP 替换 growth function：

```text
features = [
    K_A * A,
    K_R * R,
    K_O * O,
    K_T * T,
    ∇R,
    ∇O,
    A,
    H
]

Δstate = MLPθ(features)
state_next = state + dt * Δstate
```

### 优点

```text
保留 Lenia 的连续场 inductive bias
增加 NCA 的表达能力
参数量仍然小
更适合 BPTT
```

### 论文叙事

可以称为：

```text
neural local physics for continuous cellular life
```

而不是直接说“我们做了 NCA”。

---

## 22. Rule Level H：local genotype / 多物种局部规则

### 动机

如果整个世界只有一套全局规则，多个物种共存很难。Flow Lenia 的重要思想之一是参数局部化：规则参数跟随物质移动。

### 设计

```text
state = [density, hidden, genotype_vector, resource, waste]
```

局部 update 依赖 genotype：

```text
Δstate = f(local_features, genotype)
```

或：

```text
growth parameters μ, σ, kernel weights = decoder(genotype)
```

### 可能行为

```text
不同资源偏好
不同运动方式
融合 / 杂交
捕食 / 共生
局部变异
生态位分化
```

### 风险

这是长期目标。早期不要引入太多自由度，否则搜索空间极大。

---

# Part V：搜索方法路线图

## 23. 基础算法：archive + BPTT

### Archive 记录什么

```text
θ: rule parameters
init: initial body parameters
E: environment parameters
b: behavior descriptor
score: performance metrics
video/stat summary optional
```

### 每轮流程

```text
1. sample target behavior or environment challenge
2. select source θ from archive
3. mutate θ / init / environment
4. rollout T steps in PyTorch differentiable simulator
5. compute loss on final and intermediate states
6. backpropagation through time
7. optimizer step
8. evaluate over multiple random seeds
9. apply agent filters
10. store successful or interesting candidates
```

### 关键点

从 archive 找最近参数，通常不是按“历史目标距离”找，而是按“历史实际行为 descriptor”和新目标的距离找。

一次 gradient descent step 是 BPTT：跑完整个 rollout，在最终或多时刻状态上算 loss，梯度穿过所有时间步回到规则参数。

---

## 24. Loss 设计

### 基础项

```text
L_alive:
    mass in target range

L_compact:
    second moment / bounding radius

L_goal:
    final COM or density match target region

L_obstacle:
    density overlap with obstacle

L_resource:
    negative resource intake or distance to resource

L_toxic:
    overlap with toxicity

L_smooth:
    avoid high-frequency fragmentation

L_explode:
    penalize too much occupied volume
```

### 组合示例

```text
L = w_goal * L_goal
  + w_alive * L_alive
  + w_compact * L_compact
  + w_collision * L_obstacle
  + w_resource * L_resource
  + w_toxic * L_toxic
  + w_explode * L_explode
```

### 注意

不要只依赖 final target disk loss。否则容易出现：

```text
全场扩散覆盖目标
伸出细丝碰目标
牺牲主体喷射质量
爆炸式填充资源区域
```

必须配合 alive / compact / connectedness filters。

---

## 25. Agent filters

### 必要过滤器

```text
mass filter:
    mass_min < mass < mass_max

compactness filter:
    radius / second moment below threshold

connectedness filter:
    major component mass ratio > threshold

explosion filter:
    occupied volume fraction < threshold

death filter:
    final mass > threshold

long-run filter:
    after task horizon, continue simulate extra steps and remain stable
```

### 行为过滤器

```text
movement filter:
    displacement > threshold

resource filter:
    resource intake > threshold

obstacle robustness:
    success over N obstacle seeds

recovery filter:
    after damage, mass and compactness recover
```

这些 filters 是论文实验可信度的关键。否则 reviewer 会认为只是 loss hacking。

---

## 26. Behavior descriptor 设计

推荐从低维到高维：

```text
b0 = [mass, compactness]
b1 = [mass, compactness, COM_x, COM_y, COM_z]
b2 = [final COM, mean velocity, resource intake, obstacle overlap]
b3 = [trajectory PCA / Fourier features]
b4 = [multi-agent distance, resource share, coexistence time]
```

Archive 的目标不是只保存最优，而是保存行为空间中多样的可行个体。

---

## 27. Evaluation protocol

每个候选不能只在一个环境 seed 上看结果。需要：

```text
train seeds: optimization 用
validation seeds: curriculum selection 用
test seeds: final reporting 用
OOD seeds: 更大尺度、更密障碍、不同资源布局、动态扰动
```

评价指标：

```text
success rate
survival rate
mean resource intake
mean obstacle overlap
mean time-to-goal
robustness under damage
scale transfer performance
long-horizon stability
```

---

# Part VI：计算效率与工程路线

## 28. 双后端架构

### PyTorch backend

用于：

```text
可微搜索
BPTT
低分辨率实验
快速 rule iteration
metric / loss computation
```

### C++ / CUDA backend

用于：

```text
实时渲染
高分辨率展示
交互式观察
长时间 rollout
paper video / demo
```

### 共享 spec

必须维护一个 shared simulation spec：

```text
params schema
channel definitions
kernel definitions
update equations
environment field formats
random seed conventions
```

推荐目录：

```text
shared/
    volenia_spec.yaml
    params_schema.json
    environment_schema.json

python/
    volenia_diff/
        simulator_torch.py
        objectives.py
        search_archive.py
        environment_grammar.py
        export_params.py

cpp/
    simulator_cuda/
    renderer/
    import_params/
```

---

## 29. 分辨率策略

训练阶段：

```text
32^3: debug, rapid search
64^3: main differentiable experiments
128^3: selected validation
```

展示阶段：

```text
128^3 / 256^3 / higher: C++ real-time rendering
```

不要一开始在 256^3 上做 BPTT。显存和时间会很快失控。

---

## 30. PyTorch 性能优化

### 基础

```text
use torch.fft for convolution
batch multiple environments if memory allows
use mixed precision cautiously
avoid unnecessary Python loops inside step
cache static environment FFTs / SDFs
```

### BPTT 显存

```text
short horizon first: 16 / 32 / 64 steps
truncated BPTT for long tasks
gradient checkpointing
loss at sparse timepoints
low-res optimization → high-res validation
```

### 什么时候写 custom CUDA op

只有当满足以下条件才考虑：

```text
PyTorch version already produces publishable behaviors
training speed is the main bottleneck
update rule has stabilized
backward math is clear
```

早期不建议直接写 PyTorch C++/CUDA custom backward。机会成本太高。

---

## 31. Procedural environment cache

复杂 3D geometry 不应每次训练实时生成。推荐：

```text
1. offline generate environment specs
2. voxelize / SDF compute
3. save low-res and high-res versions
4. training samples from cached tensors
5. curriculum mutates parametric specs, then compiles periodically
```

缓存格式：

```text
.npz / .pt:
    obstacle: [D,H,W]
    sdf: [D,H,W]
    resource: [D,H,W]
    toxicity: [D,H,W]
    flow: [3,D,H,W]
    metadata.json
```

---

# Part VII：推荐实施阶段

## Phase 0：整理 spec 与一致性测试

目标：让 Python 和 C++ 对同一规则定义一致。

任务：

```text
定义 channel schema
定义 param schema
实现 PyTorch 2D/3D minimal Lenia step
从 C++ 导入/导出参数
同一初始状态跑 1/5/10 steps 比较误差
```

产出：

```text
volenia_spec.yaml
unit tests
one or two matched presets
```

---

## Phase 1：2D Sensorimotor Lenia 复现

目标：完全理解 archive + BPTT + obstacle curriculum。

任务：

```text
复现 2D density + obstacle channel
实现 random circular obstacles
实现 final target disk loss
实现 archive source selection
实现 inner SGD / Adam BPTT
实现 alive / compact filters
```

产出：

```text
moving patterns
obstacle robustness curves
archive visualization
```

---

## Phase 2：3D basic VolLenia agent

目标：从 2D 迁移到 3D。

任务：

```text
3D density field
target sphere loss
random sphere obstacles
alive / compact / connected filters
64^3 BPTT
C++ visualization import
```

产出：

```text
3D localized blob
3D movement
3D obstacle recovery
```

---

## Phase 3：资源与化学感受

目标：从导航坐标转向生命资源。

任务：

```text
resource channel
resource diffusion / consumption
waste channel optional
chemotaxis features
resource-seeking curriculum
resource depletion environment
```

产出：

```text
agent moves to resource
resource depleted then explores new patch
survival/resource curves
```

---

## Phase 4：复杂几何环境

目标：从随机球障碍转向结构化 3D 环境。

任务：

```text
wall / gap / tunnel / chamber grammar
SDF-based obstacle fields
static environment cache
Infinigen / ProcTHOR-like layout conversion prototype
curriculum over gap width / room complexity
```

产出：

```text
pass through tunnel
find gap in wall
navigate chamber-to-chamber
```

---

## Phase 5：neural local physics / anisotropic perception

目标：让规则表达能力跟上环境复杂度。

任务：

```text
multi-channel hidden state
small MLP growth function
learned kernel weights
directional kernel basis
optional local orientation field
ablation: parametric vs neural growth
```

产出：

```text
better obstacle navigation
better resource seeking
qualitative new behaviors
quantitative ablation
```

---

## Phase 6：multi-agent ecology

目标：从单体 agent 转向互动生态。

任务：

```text
multiple initial agents
shared resource
agent-agent overlap interactions
waste/resource cycles
local genotype optional
predator/prey or symbiosis toy tasks
```

产出：

```text
competition
coexistence
resource partitioning
simple predator-prey dynamics
```

这适合作为第二篇或长期系统目标，不一定放进第一篇主线。

---

# Part VIII：论文贡献可能形态

## 32. 最稳论文故事

```text
VolLenia: Differentiable 3D Continuous Cellular Automata for Discovering Environment-Coupled Volumetric Agents
```

贡献：

```text
1. 3D differentiable Lenia/continuous CA simulator
2. archive + BPTT + curriculum search for volumetric agents
3. obstacle/resource/SDF environment channels
4. environment progression from survival to navigation/resource seeking
5. C++/CUDA real-time volumetric visualization backend
```

## 33. 更 graphics 的故事

```text
VolLenia as an interactive 3D self-organizing volumetric primitive
```

强调：

```text
real-time volume rendering
interactive perturbation
procedural environment fields
volumetric living materials / agents
```

但需要避免只变成“好看 demo”。必须有 quantitative tasks。

## 34. 更 ALife 的故事

```text
Embodied agency emerging from differentiable 3D continuous cellular fields
```

强调：

```text
self-maintenance
sensorimotor coupling
resource metabolism
environment pressure
multi-agent interactions
```

需要更强行为分析和长期 stability。

---

# Part IX：风险与对策

## 35. 风险：搜索只找到 loss hacking

对策：

```text
agent filters
multi-seed evaluation
long-horizon stability
compactness / connectedness
OOD environment tests
human qualitative inspection only as supplement
```

## 36. 风险：3D BPTT 太慢

对策：

```text
start 32^3 / 64^3
short horizon curriculum
gradient checkpointing
low-res search high-res validation
cache environments
PyTorch first, custom CUDA later
```

## 37. 风险：环境太复杂，agent 全死

对策：

```text
manual staircase first
adaptive difficulty
POET-like not-too-easy-not-too-hard criterion
archive source selection
start from known alive presets
```

## 38. 风险：规则太简单，无法处理复杂环境

对策：

```text
multi-channel hidden state
resource and chemical sensing
small MLP growth
anisotropic kernel basis
local genotype only after simple tasks work
```

## 39. 风险：论文定位不清

对策：

始终围绕这个问题：

```text
能否通过可微搜索和环境课程，在 3D 连续 CA 中发现具有自维护、导航、避障和资源交互能力的 volumetric agents？
```

不要把论文写成：

```text
我们做了 3D Lenia，看起来很酷。
```

而要写成：

```text
我们提出一个可微分 3D continuous CA agent discovery framework，
并系统展示环境课程如何诱导越来越复杂的类生命行为。
```

---

# Part X：参考系统与用途索引

下面不是完整文献综述，而是后续实现时的参考地图。

## Lenia / CA / ALife

```text
Lenia:
    continuous CA, soft lifeforms, parameterized kernels/growth

Flow Lenia:
    mass conservation, parameter localization, multi-species direction

Sensorimotor Lenia:
    differentiable Lenia, IMGEP, BPTT, random obstacles, robust moving agents

Growing NCA:
    hidden channels, local neural update, morphogenesis, regeneration
```

## 自动课程 / 环境生成

```text
POET:
    paired environment-agent evolution, stepping stones, transfer

PAIRED / UED:
    regret-based environment generation, solvable but challenging tasks

Procgen:
    procedural diversity for generalization

OpenAI Hide-and-Seek:
    multi-agent autocurriculum and emergent tool-like behavior
```

## 多 agent / 社会困境

```text
Melting Pot:
    social dilemmas, resource sharing, cooperation/competition tests

SocialJax:
    efficient JAX social dilemma environments
```

## 3D / embodied / procedural environments

```text
Infinigen:
    photorealistic procedural natural worlds, useful for geometry extraction

Infinigen Indoors:
    indoor layout and constraint-based arrangement

Infinigen-Sim:
    articulated objects and simulation-ready assets

ProcTHOR / AI2-THOR:
    procedural indoor embodied AI scenes and navigation tasks

Habitat:
    embodied AI simulator and task API

BEHAVIOR / OmniGibson:
    complex household activity and object interaction task inspiration
```

## Voxel body / morphology optimization

```text
EvoGym:
    soft voxel robot body-control co-optimization, terrain/task curriculum inspiration
```

---

# 结论

VolLenia 的核心机会不是“3D Lenia 更好看”，而是：

> **把 Lenia 的连续生命动力学、NCA 的可训练局部规则、ALife 的环境压力、RL 的自动课程和 3D procedural 几何结合起来，形成一个可搜索的 3D artificial-life agent discovery framework。**

最现实的路径是：

```text
1. PyTorch 复现 2D Sensorimotor Lenia 式 archive + BPTT
2. 迁移到 3D basic VolLenia obstacle navigation
3. 加入 resource / toxicity / chemical sensing
4. 加入结构化 3D environment grammar
5. 加入 neural growth / anisotropic perception
6. 最后探索 multi-agent ecology
```

短期第一篇论文不需要一次完成生态系统。只要能严谨展示：

```text
3D localized volumetric agents
+ 可微搜索
+ 障碍/资源环境课程
+ quantitative robustness/generalization
+ 实时体渲染展示
```

就已经是一个足够清晰、有差异化的研究方向。
