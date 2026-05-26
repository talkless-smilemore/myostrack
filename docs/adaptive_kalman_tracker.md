# 自适应卡尔曼 + 自适应搜索因子 — 反无人机高速跟踪方案

## 一、问题背景

反无人机跟踪的核心难点之一是：**目标尺寸极小（10-30px）且运动速度快**。

现有 OSTrack 跟踪器使用"上一帧位置"作为搜索中心，配合收紧的搜索因子（3.6），搜索半边长仅有 18-54 像素。当无人机帧间位移超过搜索半边长时，目标将完全脱离搜索区域，导致跟踪失败。

本文档提出一个**自适应卡尔曼滤波 + 动态搜索因子**方案，利用运动估计自动调整搜索策略。

---

## 二、定量分析：搜索区域的运动容忍上限

> 以下数据基于对 3 个 Anti-UAV 数据集共 **610 个序列、600,937 帧帧间转移** 的统计分析。
> 统计脚本：`tools/analyze_anti_uav_motion.py`

### 2.1 搜索区域几何

```
OSTrack UAV 默认配置:
  搜索尺寸 = 256×256 px, stride = 16
  特征图   = 16×16 cells
  搜索因子 = 3.6

搜索裁剪边长 = √(W×H) × sf
搜索半边长   = √(W×H) × sf / 2
```

### 2.2 实测目标尺寸与帧间位移

| 数据集 | 序列 | 帧转移 | 中位尺寸 | 位移中位 | 位移 P90 | 位移 P95 | 位移 P99 |
|--------|------|--------|---------|---------|---------|---------|---------|
| anti_uav | 320 | 289,975 | 51.5 px | 5.0 px | 19.5 px | 31.1 px | 106 px |
| anti_uav410 | 140 | 147,656 | 29.8 px | 2.5 px | 9.0 px | 15.0 px | 46 px |
| anti_uav600 | 150 | 163,306 | 24.1 px | 2.0 px | 8.5 px | 14.5 px | 41 px |

### 2.3 搜索因子覆盖率

| 搜索因子 | anti_uav | anti_uav410 | anti_uav600 |
|---------|----------|-------------|-------------|
| sf=2.5 | 98.26% | 98.17% | 97.50% |
| **sf=3.6** | **98.94%** | **99.14%** | **98.66%** |
| sf=4.0 | 99.11% | 99.32% | 98.92% |
| sf=5.0 | 99.38% | 99.61% | 99.38% |
| sf=6.0 | 99.54% | 99.79% | 99.62% |

### 2.4 理论所需搜索因子 (`sf = 2×disp / √(w×h)`)

| 数据集 | P50 所需 | P90 所需 | P95 所需 | P99 所需 |
|--------|---------|---------|---------|---------|
| anti_uav | 0.21 | 0.66 | 1.02 | 3.74 |
| anti_uav410 | 0.19 | 0.72 | 1.25 | 3.37 |
| anti_uav600 | 0.21 | 0.89 | 1.55 | 4.14 |

**结论**：
- 搜索因子 **3.6 在实际数据中已经覆盖了 ~99%** 的帧间转移，并非"仅覆盖慢巡"
- P95 所需搜索因子仅为 1.0-1.55，远低于 3.6
- 自适应搜索因子的价值在于覆盖 P99 以上的**极端帧**（占比 < 1%），而非常态巡航

---

## 三、自适应卡尔曼滤波器设计

### 3.1 核心思路

传统 CV 卡尔曼的线性高斯假设在无人机机动时被打破。自适应卡尔曼通过**innovation-based Q 调整**来提高对机动的响应能力：

- **正常平飞**：Q 保持基准值，提供平滑滤波
- **机动飞行**（急转、加减速）：Q 自动放大，让滤波器快速跟上
- **长时稳定**：Q 逐渐衰减回基准

### 3.2 完整实现

```python
"""
lib/test/tracker/kalman_filter.py
自适应卡尔曼滤波器 — 针对反无人机快速机动场景
"""

import torch
import numpy as np


class AdaptiveKalmanFilter:
    """自适应卡尔曼滤波器（CV 模型 + Innovation-based Q 调整）

    状态: x = [cx, cy, w, h, vx, vy]^T      (6维)
    观测: z = [cx, cy, w, h]^T               (4维)

    自适应机制:
      当 observation innovation（新息）的 Mahalanobis 距离超过阈值时，
      临时增大过程噪声 Q，使滤波器更信任观测而非模型预测。
      当新息恢复正常后，Q 逐渐衰减回基准值。
    """

    def __init__(self,
                 dt: float = 1.0,
                 base_process_noise: float = 0.01,
                 velocity_noise_scale: float = 10.0,
                 size_noise_scale: float = 2.0,
                 base_observation_noise: float = 5.0,
                 adapt_threshold: float = 5.0,
                 adapt_gain: float = 2.0,
                 decay_rate: float = 0.95,
                 max_q_scale: float = 10.0):
        """
        Args:
            dt: 帧间时间间隔（帧率归一化后通常为 1.0）
            base_process_noise: 基础过程噪声（位置项）
            velocity_noise_scale: 速度噪声相对于位置噪声的倍率
            size_noise_scale: 尺寸噪声相对于位置噪声的倍率
            base_observation_noise: 基础观测噪声
            adapt_threshold: 触发自适应 Q 调整的 Mahalanobis 距离阈值
            adapt_gain: Q 放大的倍增因子
            decay_rate: Q 衰减回基准值的速率（每帧乘以该因子）
            max_q_scale: Q 放大倍率的上限
        """
        self.dt = dt

        # ---- 状态转移矩阵 F (6×6) ----
        # 恒定速度模型: cx += vx*dt, cy += vy*dt
        self.F = torch.eye(6, dtype=torch.float32)
        self.F[0, 4] = dt
        self.F[1, 5] = dt

        # ---- 观测矩阵 H (4×6) ----
        self.H = torch.zeros(4, 6, dtype=torch.float32)
        self.H[0, 0] = 1.0   # z_cx = cx
        self.H[1, 1] = 1.0   # z_cy = cy
        self.H[2, 2] = 1.0   # z_w  = w
        self.H[3, 3] = 1.0   # z_h  = h

        # ---- 基础过程噪声 Q_base (6×6) ----
        # 位置噪声小（假设位置通过积分确定）、速度噪声大（允许机动）
        self._Q_base = torch.diag(torch.tensor([
            base_process_noise,
            base_process_noise,
            base_process_noise * size_noise_scale,
            base_process_noise * size_noise_scale,
            base_process_noise * velocity_noise_scale,
            base_process_noise * velocity_noise_scale,
        ], dtype=torch.float32))
        self.Q = self._Q_base.clone()
        self._q_scale = 1.0  # 当前的 Q 放缩因子

        # ---- 观测噪声 R (4×4) ----
        self.R = torch.diag(torch.tensor([
            base_observation_noise,
            base_observation_noise,
            base_observation_noise,
            base_observation_noise,
        ], dtype=torch.float32))

        # ---- 自适应参数 ----
        self.adapt_threshold = adapt_threshold
        self.adapt_gain = adapt_gain
        self.decay_rate = decay_rate
        self.max_q_scale = max_q_scale

        # ---- 滤波器状态 ----
        self.x = None   # (6,)
        self.P = None   # (6,6)

        # ---- 状态历史（用于诊断） ----
        self._last_innovation = None
        self._last_mahalanobis = 0.0

    # =================================================================
    # 公共接口
    # =================================================================

    def init(self, cx: float, cy: float, w: float, h: float):
        """用第一帧的目标框初始化滤波器。

        Args:
            cx, cy: 目标中心坐标（图像坐标系，像素）
            w, h:   目标宽高（像素）
        """
        self.x = torch.tensor(
            [cx, cy, w, h, 0.0, 0.0], dtype=torch.float32)
        # 协方差矩阵初始化为大值 → 前几帧以观测为主，快速收敛速度
        self.P = torch.eye(6, dtype=torch.float32) * 100.0
        self._q_scale = 1.0
        self.Q = self._Q_base.clone()

    def predict(self):
        """先验预测步骤。返回预测的位置 + 尺寸 [cx, cy, w, h]."""
        if self.x is None:
            raise RuntimeError("KalmanFilter not initialized. Call init() first.")
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].clone()

    def update(self, z_cx: float, z_cy: float, z_w: float, z_h: float):
        """用模型输出的观测更新滤波器。

        Args:
            z_cx, z_cy: 模型预测的目标中心（图像坐标，像素）
            z_w, z_h:   模型预测的宽高（像素）
        """
        z = torch.tensor([z_cx, z_cy, z_w, z_h], dtype=torch.float32)
        y = z - self.H @ self.x            # (4,) 新息（innovation）
        S = self.H @ self.P @ self.H.T + self.R  # (4,4) 新息协方差

        # 卡尔曼增益
        K = self.P @ self.H.T @ torch.linalg.inv(S)

        # 状态更新
        self.x = self.x + K @ y
        self.P = (torch.eye(6) - K @ self.H) @ self.P

        # ---- 自适应 Q 调整 ----
        self._adapt_q(y, S)

        self._last_innovation = y
        return self.x[:4].clone()

    def update_with_confidence(self, z_cx, z_cy, z_w, z_h, confidence: float,
                                conf_threshold: float = 0.3):
        """置信度门控的更新：仅在模型置信度 > 阈值时才作为观测输入。

        Args:
            confidence: 模型 score_map 的最大值 [0, 1]
            conf_threshold: 低于此阈值时跳过更新
        Returns:
            (filtered_box, used_observation): 滤波后的框 + 是否使用了观测
        """
        if confidence > conf_threshold:
            filtered = self.update(z_cx, z_cy, z_w, z_h)
            used = True
        else:
            # 低置信度：跳过更新，直接返回预测值
            filtered = self.x[:4].clone()
            used = False
        return filtered, used

    def get_state(self):
        """获取当前滤波后的状态 [cx, cy, w, h]."""
        if self.x is None:
            raise RuntimeError("KalmanFilter not initialized.")
        return self.x[:4].clone()

    def get_velocity(self):
        """获取估计速度 [vx, vy]."""
        if self.x is None:
            return torch.tensor([0.0, 0.0])
        return self.x[4:6].clone()

    @property
    def estimated_speed(self) -> float:
        """估计速度大小（像素/帧）."""
        vx, vy = self.get_velocity()
        return float(torch.sqrt(vx ** 2 + vy ** 2).item())

    # =================================================================
    # 自适应 Q 调整
    # =================================================================

    def _adapt_q(self, innovation: torch.Tensor, innovation_cov: torch.Tensor):
        """基于新息的 Mahalanobis 距离自适应调整过程噪声 Q。

        原理:
          Mahalanobis 距离 = sqrt(y^T * S^{-1} * y)
          如果距离 > threshold → 模型预测与实际观测不匹配 → 放大 Q
          如果距离 < threshold → 模型预测吻合 → 衰减 Q
        """
        # Mahalanobis 距离平方
        d2 = float(innovation @ torch.linalg.inv(innovation_cov) @ innovation)
        self._last_mahalanobis = np.sqrt(max(0.0, d2))

        if self._last_mahalanobis > self.adapt_threshold:
            # 机动检测：放大过程噪声
            self._q_scale = min(self._q_scale * self.adapt_gain, self.max_q_scale)
        else:
            # 平稳飞行：衰减过程噪声回基准
            self._q_scale = max(self._q_scale * self.decay_rate, 1.0)

        self.Q = self._Q_base * self._q_scale

    def get_filtered_box_xywh(self):
        """以 [x1, y1, w, h] 格式返回当前滤波框."""
        cx, cy, w, h = self.get_state()
        return [float(cx - w / 2), float(cy - h / 2), float(w), float(h)]
```

### 3.3 自适应行为示意图

```
Mahalanobis 距离 (新息异常程度):
    ^
 10 |                    ● ← 急转弯，Q × 2
    |                   / \
  6 |──────────────────/───\───────── 自适应阈值
    |                 /     \
  2 |     ●--●--●-●-/---●--●--●--●   ← 平稳飞行，Q 衰减至 1.0
    |    /
  0 |──●───────────────────────────→ 帧
    |
    Q_scale:
      平稳: 1.0 → 平滑轨迹
      机动: 2~8 → 快速响应，减少滞后
```

---

## 四、自适应搜索因子

### 4.1 核心公式

基于物理上"搜索区域必须覆盖帧间位移"的约束：

```
target_diag     = max(1.0, √(w² + h²))        (目标对角线像素)
predicted_disp  = estimated_speed × dt         (卡尔曼预测帧间位移)
required_sf     = 2 × predicted_disp / target_diag   (所需最小搜索因子)
safe_sf         = SAFETY_MARGIN × required_sf        (加安全容限, 默认 1.5×)
sf              = clamp(safe_sf, SF_MIN, SF_MAX)     (钳位)
```

即搜索因子直接由物理约束推导，而非启发式加法。

### 4.2 基于真实数据的参数

| 场景 | 速度 | 目标尺寸 | 所需 sf | 自适应输出 | 说明 |
|------|------|---------|--------|-----------|------|
| 悬停 | ~0 px/f | 任意 | ~0 | sf=3.6(基准) | 绝大多数帧 |
| 常态 | 5 px/f | 30 px | 0.33 | sf=3.6(基准) | 中位场景 |
| P90 | 20 px/f | 50 px | 0.80 | sf=3.6(基准) | anti_uav 的 P90 |
| P95 | 31 px/f | 50 px | 1.24 | sf=3.6(基准) | anti_uav 的 P95 |
| P99 | 106 px/f | 50 px | 4.24 | **sf=6.0**(上限) | 极端帧，扩到最大 |
| P99 | 46 px/f | 25 px | 3.68 | sf=5.5 | anti_uav410 的 P99 |
| P99 | 41 px/f | 24 px | 3.42 | sf=5.1 | anti_uav600 的 P99 |
| 极端 | 541 px/f | 13 px | >80 | sf=6.0(上限) | 数据标注噪声 |

**实际数据验证**：
- P50~P95 所需 sf=0.2~1.5，全部 < 3.6 → **自适应机制在常态下不触发扩大**
- P99 所需 sf=3.4~4.1 → 仅在极端帧触发扩大，占比 < 1%
- 触发扩大时 CE (70% 保留率) 有效控制 token 数量

### 4.3 实现

```python
def compute_adaptive_search_factor(
    estimated_speed: float,      # 像素/帧
    target_w: float,             # 像素
    target_h: float,             # 像素
    base_factor: float = 3.6,
    min_factor: float = 2.5,
    max_factor: float = 5.0,
    safety_margin: float = 1.5,
) -> float:
    """根据卡尔曼估计速度自适应计算搜索因子。

    物理原理:
      搜索半边长 = sqrt(w×h) × sf / 2
      要覆盖预测位移 → sf = 2 × predicted_disp / sqrt(w×h)
      再乘以安全容限(safety_margin)确保预测误差不导致目标出界。

    基于 Anti-UAV 真实数据 (610序列, 600K+帧转移):
      - sf=3.6 已覆盖 98.7~99.1% 的帧间转移
      - 仅 P99 以上的极端帧需要自适应扩大
    """
    target_diag = max(1.0, math.sqrt(target_w ** 2 + target_h ** 2))
    predicted_disp = estimated_speed  # dt=1 时

    # 物理所需最小搜索因子
    required = 2.0 * predicted_disp / target_diag
    safe_sf = safety_margin * required

    return max(min_factor,
               min(max_factor,
                   max(safe_sf, base_factor)))
```

---

## 五、集成到 OSTrack 跟踪器

### 5.1 完整修改后的 `ostrack.py`

```python
"""
lib/test/tracker/ostrack.py  (主要修改部分)

修改要点:
  1. 引入 AdaptiveKalmanFilter 替代"上一帧位置"作为搜索中心
  2. 引入 compute_adaptive_search_factor 动态调整搜索区域大小
  3. 置信度门控: 低置信度时跳过卡尔曼更新
  4. 丢失回退: 连续低置信度 → 扩大搜索区域做重检测
"""

import math
import torch
import numpy as np
import cv2
import os

from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.test.tracker.kalman_filter import AdaptiveKalmanFilter


class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        checkpoint = torch.load(self.params.checkpoint, map_location='cpu',
                                weights_only=False)
        network.load_state_dict(checkpoint['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = (self.cfg.TEST.SEARCH_SIZE //
                        self.cfg.MODEL.BACKBONE.STRIDE)
        self.output_window = hann2d(
            torch.tensor([self.feat_sz, self.feat_sz]).long(),
            centered=True).cuda()

        # ---- 新增: 自适应卡尔曼滤波器 ----
        kf_cfg = getattr(self.cfg, 'KALMAN_FILTER', {})
        self.kf = AdaptiveKalmanFilter(
            dt=getattr(kf_cfg, 'DT', 1.0),
            base_process_noise=getattr(kf_cfg, 'BASE_PROCESS_NOISE', 0.01),
            velocity_noise_scale=getattr(kf_cfg, 'VELOCITY_NOISE_SCALE', 10.0),
            base_observation_noise=getattr(kf_cfg, 'BASE_OBS_NOISE', 5.0),
            adapt_threshold=getattr(kf_cfg, 'ADAPT_THRESHOLD', 5.0),
            adapt_gain=getattr(kf_cfg, 'ADAPT_GAIN', 2.0),
            max_q_scale=getattr(kf_cfg, 'MAX_Q_SCALE', 10.0),
        )
        self.kf_conf_threshold = getattr(kf_cfg, 'CONF_THRESHOLD', 0.3)

        # ---- 新增: 自适应搜索因子参数 ----
        self.base_search_factor = self.params.search_factor  # 3.6
        self.min_search_factor = getattr(kf_cfg, 'MIN_SEARCH_FACTOR', 2.5)
        self.max_search_factor = getattr(kf_cfg, 'MAX_SEARCH_FACTOR', 5.0)
        self.safety_margin = getattr(kf_cfg, 'SAFETY_MARGIN', 1.5)

        # ---- 新增: 丢失回退 ----
        self.low_conf_counter = 0
        self.max_low_conf_frames = getattr(kf_cfg, 'MAX_LOW_CONF_FRAMES', 5)

        # ---- 调试 ----
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)

        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    # =================================================================
    # 初始化
    # =================================================================

    def initialize(self, image, info: dict):
        """用第一帧初始化跟踪器和卡尔曼滤波器。"""
        # ---- 模板提取 (不变) ----
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image, info['init_bbox'],
            self.params.template_factor,
            output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(
                info['init_bbox'], resize_factor,
                template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(
                self.cfg, 1, template.tensors.device, template_bbox)

        # ---- 卡尔曼初始化 ----
        bbox = info['init_bbox']
        cx_init = bbox[0] + bbox[2] / 2.0
        cy_init = bbox[1] + bbox[3] / 2.0
        self.kf.init(cx_init, cy_init, bbox[2], bbox[3])

        # ---- 保存状态 ----
        self.state = info['init_bbox']
        self.frame_id = 0
        self.low_conf_counter = 0

        if self.save_all_boxes:
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    # =================================================================
    # 跟踪主循环
    # =================================================================

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        # ---- Step 1: 卡尔曼预测 ----
        pred_state = self.kf.predict()
        pred_cx, pred_cy, pred_w, pred_h = pred_state

        # ---- Step 2: 自适应搜索因子 ----
        est_speed = self.kf.estimated_speed
        adaptive_sf = self._compute_adaptive_search_factor(
            est_speed, pred_w, pred_h)

        # ---- Step 3: 以预测位置 + 自适应因子裁剪搜索区域 ----
        kf_box = [pred_cx - pred_w / 2.0,
                  pred_cy - pred_h / 2.0,
                  max(pred_w, 1.0),
                  max(pred_h, 1.0)]
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, kf_box, adaptive_sf,
            output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        # ---- Step 4: 模型前向 ----
        with torch.no_grad():
            out_dict = self.network.forward(
                template=self.z_dict1.tensors,
                search=search.tensors,
                ce_template_mask=self.box_mask_z)

        # ---- Step 5: 解码预测框 ----
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(
            response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) *
                    self.params.search_size / resize_factor).tolist()
        # pred_box: (cx, cy, w, h) 归一化 [0, 1]

        # ---- Step 6: 将预测框反算回图像坐标 ----
        pred_cx_img = (pred_box[0] +
                       (pred_cx - 0.5 * self.params.search_size / resize_factor))
        pred_cy_img = (pred_box[1] +
                       (pred_cy - 0.5 * self.params.search_size / resize_factor))
        pred_w_img = pred_box[2]
        pred_h_img = pred_box[3]

        # ---- Step 7: 置信度门控卡尔曼更新 ----
        confidence = float(pred_score_map.max().item())

        if confidence > self.kf_conf_threshold:
            # 高置信度: 用模型输出更新卡尔曼
            self.kf.update(pred_cx_img, pred_cy_img, pred_w_img, pred_h_img)
            self.low_conf_counter = 0

            # 使用卡尔曼滤波后的状态
            filtered = self.kf.get_state()
            fc, fy, fw, fh = filtered
            self.state = clip_box(
                [float(fc - fw / 2), float(fy - fh / 2), float(fw), float(fh)],
                H, W, margin=10)
        else:
            # 低置信度: 跳过更新，用预测维持
            self.low_conf_counter += 1
            self.state = clip_box(
                [float(pred_cx - pred_w / 2), float(pred_cy - pred_h / 2),
                 float(pred_w), float(pred_h)], H, W, margin=10)

        # ---- Step 8: 丢失回退 ----
        if self.low_conf_counter > self.max_low_conf_frames:
            # 长时间低置信度: 扩大搜索做一次全局尝试
            self.state = self._attempt_recovery(image, H, W)

        # ---- 调试可视化 (不变) ----
        if self.debug:
            self._debug_visualize(image, info, out_dict, x_patch_arr)

        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(
                pred_boxes * self.params.search_size / resize_factor,
                resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()
            return {"target_bbox": self.state, "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    # =================================================================
    # 辅助方法
    # =================================================================

    def _compute_adaptive_search_factor(self, est_speed, target_w, target_h):
        """根据卡尔曼估计速度，基于物理约束计算搜索因子。

        sf = 2 * predicted_disp / target_diag (物理最小)
        再乘以 safety_margin 作为安全容限。
        """
        target_diag = max(1.0, math.sqrt(target_w ** 2 + target_h ** 2))
        predicted_disp = est_speed
        required = 2.0 * predicted_disp / target_diag
        safe_sf = self.safety_margin * required

        return max(self.min_search_factor,
                   min(self.max_search_factor,
                       max(safe_sf, self.base_search_factor)))

    def _attempt_recovery(self, image, H, W):
        """丢失回退: 用更大的搜索因子做最后一次尝试"""
        # 回退到卡尔曼预测位置
        pred = self.kf.get_state()
        recovery_box = [float(pred[0] - pred[2] / 2),
                        float(pred[1] - pred[3] / 2),
                        float(pred[2]), float(pred[3])]

        # 扩大搜索因子
        x_patch, resize_factor, x_mask = sample_target(
            image, recovery_box, self.max_search_factor,
            output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch, x_mask)

        with torch.no_grad():
            out_dict = self.network.forward(
                template=self.z_dict1.tensors,
                search=search.tensors,
                ce_template_mask=self.box_mask_z)

        score_map = out_dict['score_map']
        conf = float(score_map.max().item())

        if conf > self.kf_conf_threshold * 0.7:  # 放宽阈值
            # 恢复成功
            response = self.output_window * score_map
            pred_boxes = self.network.box_head.cal_bbox(
                response, out_dict['size_map'], out_dict['offset_map'])
            pred_box = (pred_boxes.mean(dim=0) *
                        self.params.search_size / resize_factor).tolist()

            pred_cx = (pred_box[0] + (pred[0] -
                       0.5 * self.params.search_size / resize_factor))
            pred_cy = (pred_box[1] + (pred[1] -
                       0.5 * self.params.search_size / resize_factor))

            self.kf.update(float(pred_cx), float(pred_cy),
                           float(pred_box[2]), float(pred_box[3]))
            self.low_conf_counter = 0
            filtered = self.kf.get_state()
            fc, fy, fw, fh = filtered
            return clip_box([float(fc - fw / 2), float(fy - fh / 2),
                             float(fw), float(fh)], H, W, margin=10)

        # 恢复失败，继续用卡尔曼预测
        return clip_box(recovery_box, H, W, margin=10)

    def _debug_visualize(self, image, info, out_dict, x_patch_arr):
        """调试可视化。"""
        if not self.use_visdom:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)),
                          (int(x1 + w), int(y1 + h)),
                          color=(0, 0, 255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        else:
            self.visdom.register(
                (image, info['gt_bbox'].tolist(), self.state),
                'Tracking', 1, 'Tracking')
            self.visdom.register(
                torch.from_numpy(x_patch_arr).permute(2, 0, 1),
                'image', 1, 'search_region')
            self.visdom.register(
                torch.from_numpy(self.z_patch_arr).permute(2, 0, 1),
                'image', 1, 'template')

            pred_score_map = out_dict['score_map']
            self.visdom.register(
                pred_score_map.view(self.feat_sz, self.feat_sz),
                'heatmap', 1, 'score_map')

            if ('removed_indexes_s' in out_dict and
                    out_dict['removed_indexes_s']):
                removed_indexes_s = out_dict['removed_indexes_s']
                removed_indexes_s = [ri.cpu().numpy()
                                     for ri in removed_indexes_s]
                masked_search = gen_visualization(x_patch_arr,
                                                  removed_indexes_s)
                self.visdom.register(
                    torch.from_numpy(masked_search).permute(2, 0, 1),
                    'image', 1, 'masked_search')

            while self.pause_mode:
                if self.step:
                    self.step = False
                    break

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = (self.state[0] + 0.5 * self.state[2],
                            self.state[1] + 0.5 * self.state[3])
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        return [cx + (cx_prev - half_side) - 0.5 * w,
                cy + (cy_prev - half_side) - 0.5 * h,
                w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = (self.state[0] + 0.5 * self.state[2],
                            self.state[1] + 0.5 * self.state[3])
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h],
                           dim=-1)


def get_tracker_class():
    return OSTrack
```

---

## 六、配置项

在 YAML 中新增 KALMAN_FILTER 配置块：

```yaml
# experiments/ostrack/vitb_384_mae_ce_32x4_ep300_uav_oplora_kf.yaml

KALMAN_FILTER:
  ENABLE: true
  DT: 1.0                       # 帧间间隔

  # 过程噪声
  BASE_PROCESS_NOISE: 0.01      # 位置基础噪声
  VELOCITY_NOISE_SCALE: 10.0    # 速度噪声倍率（>1 → 允许机动）
  SIZE_NOISE_SCALE: 2.0         # 尺寸噪声倍率

  # 观测噪声
  BASE_OBS_NOISE: 5.0           # 观测基础噪声（越大 → 越信任预测）

  # 自适应参数
  ADAPT_THRESHOLD: 5.0          # Mahalanobis 距离阈值
  ADAPT_GAIN: 2.0               # Q 放大倍率
  DECAY_RATE: 0.95              # Q 衰减率
  MAX_Q_SCALE: 10.0             # Q 最大放大倍率

  # 置信度门控
  CONF_THRESHOLD: 0.3           # score_map 最大值低于此阈值时跳过更新

  # 自适应搜索因子
  MIN_SEARCH_FACTOR: 2.5        # 下限
  MAX_SEARCH_FACTOR: 5.0        # 上限
  SAFETY_MARGIN: 1.5            # 物理所需 sf × 安全容限 (P99 所需=3.4~4.1, ×1.5 → 触达 5.0)

  # 丢失回退
  MAX_LOW_CONF_FRAMES: 5        # 连续低置信度帧数，触发重检测
```

---

## 七、关键参数调优指南

### 7.1 最敏感的 4 个参数

| 参数 | 调大效果 | 调小效果 | 建议范围 |
|------|---------|---------|---------|
| `BASE_OBS_NOISE` | 更信任预测，轨迹更平滑，但对机动响应慢 | 更信任观测，反应快但容易抖动 | 3~10 |
| `VELOCITY_NOISE_SCALE` | 允许更大的速度变化（更敏感的机动检测） | 假设更平稳的运动 | 5~20 |
| `ADAPT_THRESHOLD` | 更难触发 Q 自适应（更保守） | 更容易触发 Q 自适应 | 3~8 |
| `SAFETY_MARGIN` | 物理所需 sf 的安全倍率，越大越保守 | 更小的搜索区域，更快，但有丢失风险 | 1.2~2.0 |

### 7.2 基于真实数据的推荐配置

以下配置基于对 610 个序列、600K+ 帧转移的统计分析。

**所有数据集的统一推荐（默认配置）：**

```yaml
KALMAN_FILTER:
  BASE_PROCESS_NOISE: 0.01
  VELOCITY_NOISE_SCALE: 10.0
  BASE_OBS_NOISE: 5.0
  ADAPT_THRESHOLD: 5.0
  CONF_THRESHOLD: 0.3
  MIN_SEARCH_FACTOR: 2.5       # 下限 (P5 所需 sf 接近 0)
  MAX_SEARCH_FACTOR: 5.0       # 上限 (P99 所需 sf=3.4~4.1, 加安全容限=5.0)
  SAFETY_MARGIN: 1.5           # 在物理所需 sf 上乘 1.5 倍安全容限
  MAX_LOW_CONF_FRAMES: 5
```

**按数据集差异化建议：**

| 参数 | anti_uav | anti_uav410 | anti_uav600 |
|------|----------|-------------|-------------|
| MAX_SEARCH_FACTOR | 5.0 | 5.0 | 6.0 |
| VELOCITY_NOISE_SCALE | 10.0 | 8.0 | 8.0 |
| 理由 | 目标更大(P50=51px), 位移极端值高(P99=106px) | 目标较小(P50=30px), 运动较平稳 | 目标最小(P50=24px), 极少数序列位移异常 |
| sf=3.6 覆盖 | 98.94% | 99.14% | 98.66% |
| sf=5.0 覆盖 | 99.38% | 99.61% | 99.38% |
| sf=6.0 覆盖 | 99.54% | 99.79% | 99.62% |

---

## 八、局限性 & 物理上限

自适应搜索因子有物理上限：

```
状况                                处理策略
──────────────────────────────────────────────────────
目标帧间位移 < 搜索半边长            正常跟踪
目标帧间位移 > 搜索半边长 (sf=6.0)   扩大搜索因子 + 卡尔曼预测
目标帧间位移 > 搜索半边长 (sf=8.0)   触发重检测 / 丢失，等待目标减速
连续丢失 > 30 帧                    放弃该轨迹，切换到目标重检测模式
```

**极端情况限制**：当无人机帧间位移超过系统物理上限时，唯一解法是提高帧率（如 60fps → 位移减半）或使用更大视场角的传感器。这不是算法层面的限制，而是**信息层面的限制**（你在搜索区域内确实没有目标）。

---

## 九、涉及的文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `lib/test/tracker/kalman_filter.py` | 新建 | `AdaptiveKalmanFilter` 类 |
| `lib/test/tracker/ostrack.py` | 修改 | 集成 KF + 自适应搜索因子 |
| `lib/config/ostrack/config.py` | 修改 | 新增 KALMAN_FILTER 配置块 |
| `experiments/ostrack/*_kf.yaml` | 新建 | 含 KF 配置的实验 YAML |
