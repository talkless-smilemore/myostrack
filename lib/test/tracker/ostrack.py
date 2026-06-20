import math

from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.kalman_filter import AdaptiveKalmanFilter
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        checkpoint = torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)
        network.load_state_dict(checkpoint['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # ---- Adaptive Kalman filter with innovation-based Q ----
        kf_cfg = getattr(self.cfg.TEST, "KALMAN_FILTER", None)
        self.kf_enable = kf_cfg is not None and getattr(kf_cfg, "ENABLE", False)
        if self.kf_enable:
            self.kf = AdaptiveKalmanFilter(
                dt=getattr(kf_cfg, "DT", 1.0),
                base_process_noise=getattr(kf_cfg, "BASE_PROCESS_NOISE", 0.01),
                velocity_noise_scale=getattr(kf_cfg, "VELOCITY_NOISE_SCALE", 10.0),
                size_noise_scale=getattr(kf_cfg, "SIZE_NOISE_SCALE", 2.0),
                base_observation_noise=getattr(kf_cfg, "BASE_OBS_NOISE", 5.0),
                adapt_threshold=getattr(kf_cfg, "ADAPT_THRESHOLD", 5.0),
                adapt_gain=getattr(kf_cfg, "ADAPT_GAIN", 2.0),
                decay_rate=getattr(kf_cfg, "DECAY_RATE", 0.95),
                max_q_scale=getattr(kf_cfg, "MAX_Q_SCALE", 10.0),
            )
            self.kf_conf_threshold = getattr(kf_cfg, "CONF_THRESHOLD", 0.3)
            self.kf_min_sf = getattr(kf_cfg, "MIN_SEARCH_FACTOR", 2.5)
            self.kf_max_sf = getattr(kf_cfg, "MAX_SEARCH_FACTOR", 5.0)
            self.kf_safety_margin = getattr(kf_cfg, "SAFETY_MARGIN", 1.5)
            self.kf_max_low_conf = getattr(kf_cfg, "MAX_LOW_CONF_FRAMES", 5)
        else:
            self.kf = None

        self.low_conf_counter = 0

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # ---- initialise Kalman filter ----
        if self.kf is not None:
            bbox = info['init_bbox']
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0
            self.kf.init(cx, cy, bbox[2], bbox[3])

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        self.low_conf_counter = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        # ---- compute search centre & search factor ----
        if self.kf is not None:
            pred_state = self.kf.predict()
            pred_cx, pred_cy, pred_w, pred_h = [float(v) for v in pred_state]
            pred_w = max(pred_w, 1.0)
            pred_h = max(pred_h, 1.0)

            # adaptive search factor
            est_speed = self.kf.estimated_speed
            sf = self._adaptive_search_factor(est_speed, pred_w, pred_h)

            search_box = [pred_cx - pred_w / 2.0, pred_cy - pred_h / 2.0,
                          pred_w, pred_h]
        else:
            search_box = self.state
            sf = self.params.search_factor
            pred_cx = None  # for map_box_back fallback

        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, search_box, sf,
            output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

        confidence = float(pred_score_map.max().item())

        # ---- map prediction back to image coordinates ----
        if self.kf is not None and pred_cx is not None:
            half_side = 0.5 * self.params.search_size / resize_factor
            pred_cx_img = pred_box[0] + (pred_cx - half_side)
            pred_cy_img = pred_box[1] + (pred_cy - half_side)
            pred_w_img = pred_box[2]
            pred_h_img = pred_box[3]
        else:
            # fallback to original map_box_back
            cx_prev = self.state[0] + 0.5 * self.state[2]
            cy_prev = self.state[1] + 0.5 * self.state[3]
            half_side = 0.5 * self.params.search_size / resize_factor
            pred_cx_img = pred_box[0] + (cx_prev - half_side)
            pred_cy_img = pred_box[1] + (cy_prev - half_side)
            pred_w_img = pred_box[2]
            pred_h_img = pred_box[3]

        # ---- state update with Kalman confidence gate ----
        if self.kf is not None:
            if confidence > self.kf_conf_threshold:
                self.kf.update(pred_cx_img, pred_cy_img, pred_w_img, pred_h_img)
                self.low_conf_counter = 0
                # use filtered state
                fc, fy, fw, fh = [float(v) for v in self.kf.get_state()]
                self.state = clip_box(
                    [fc - fw / 2.0, fy - fh / 2.0, fw, fh], H, W, margin=10)
            else:
                self.low_conf_counter += 1
                self.state = clip_box(
                    [pred_cx_img - pred_w_img / 2.0,
                     pred_cy_img - pred_h_img / 2.0,
                     pred_w_img, pred_h_img], H, W, margin=10)

            # lost-target recovery
            if self.low_conf_counter > self.kf_max_low_conf:
                self.state = self._kf_recovery_attempt(image, H, W, pred_score_map)
        else:
            self.state = clip_box(
                [pred_cx_img - pred_w_img / 2.0,
                 pred_cy_img - pred_h_img / 2.0,
                 pred_w_img, pred_h_img], H, W, margin=10)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def _adaptive_search_factor(self, est_speed: float, target_w: float, target_h: float) -> float:
        """Adaptive search factor based on estimated target speed.

        Uses the physical constraint: sf = 2 × displacement / target_diag
        with a safety margin, clamped to configured min/max.
        """
        target_diag = max(1.0, math.sqrt(target_w ** 2 + target_h ** 2))
        predicted_disp = est_speed
        required = 2.0 * predicted_disp / target_diag
        safe_sf = self.kf_safety_margin * required
        return max(self.kf_min_sf,
                   min(self.kf_max_sf,
                       max(safe_sf, self.params.search_factor)))

    def _kf_recovery_attempt(self, image, H, W, prev_score_map):
        """Widen the search region and try to re-detect the target."""
        pred = self.kf.get_state()
        fc, fy, fw, fh = [float(v) for v in pred]
        recovery_box = [fc - fw / 2.0, fy - fh / 2.0, max(fw, 1.0), max(fh, 1.0)]

        x_patch, resize_factor, x_mask = sample_target(
            image, recovery_box, self.kf_max_sf,
            output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch, x_mask)

        with torch.no_grad():
            out_dict = self.network.forward(
                template=self.z_dict1.tensors,
                search=search.tensors,
                ce_template_mask=self.box_mask_z)

        score_map = out_dict['score_map']
        conf = float(score_map.max().item())

        if conf > self.kf_conf_threshold * 0.7:
            response = self.output_window * score_map
            pred_boxes = self.network.box_head.cal_bbox(
                response, out_dict['size_map'], out_dict['offset_map'])
            pred_box = (pred_boxes.mean(dim=0) *
                        self.params.search_size / resize_factor).tolist()

            half_side = 0.5 * self.params.search_size / resize_factor
            meas_cx = pred_box[0] + (fc - half_side)
            meas_cy = pred_box[1] + (fy - half_side)

            self.kf.update(meas_cx, meas_cy, pred_box[2], pred_box[3])
            self.low_conf_counter = 0

            filtered = self.kf.get_state()
            ff = [float(v) for v in filtered]
            return clip_box(
                [ff[0] - ff[2] / 2.0, ff[1] - ff[3] / 2.0, ff[2], ff[3]],
                H, W, margin=10)

        return clip_box(recovery_box, H, W, margin=10)

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OSTrack
