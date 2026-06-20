from easydict import EasyDict as edict
import yaml

"""
Add default config for OSTrack.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
cfg.MODEL.EXTRA_MERGER = False

cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256

# MODEL.CENTER_PRIOR 鈥?Euclidean distance centre prior (Fan et al. 2026)
cfg.MODEL.CENTER_PRIOR = edict()
cfg.MODEL.CENTER_PRIOR.ENABLE = False
cfg.MODEL.CENTER_PRIOR.NUM_BINS = 16

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
# Vanilla LoRA baseline. This is the original low-rank adapter without
# orthogonal projection, gates, neuron selection, WSP, or extra regularisers.
cfg.TRAIN.LORA = edict()
cfg.TRAIN.LORA.ENABLE = False
cfg.TRAIN.LORA.RANK = 8
cfg.TRAIN.LORA.ALPHA = 8.0
cfg.TRAIN.LORA.DROPOUT = 0.0
cfg.TRAIN.LORA.TARGETS = ["qkv", "proj", "fc1", "fc2"]
cfg.TRAIN.LORA.FREEZE_BACKBONE = True
# 鈺愨晲鈺?DEPRECATED 鈥?kept only for YAML backward compat 鈺愨晲鈺?
# OPLoRA, NS-OPLoRA, and SGLoRA are superseded by UAV-WSP below.
# These config stubs exist so old experiments/*.yaml files don't crash
# config._update_config(). Do NOT use them for new experiments.
cfg.TRAIN.OPLORA = edict()
cfg.TRAIN.OPLORA.ENABLE = False
cfg.TRAIN.OPLORA.RANK = 8
cfg.TRAIN.OPLORA.TOP_K = 16
cfg.TRAIN.OPLORA.ALPHA = 8.0
cfg.TRAIN.OPLORA.TARGETS = ["qkv", "proj", "fc1", "fc2"]
cfg.TRAIN.NEURO_OPLORA = edict()
cfg.TRAIN.NEURO_OPLORA.ENABLE = False
cfg.TRAIN.NEURO_OPLORA.LAYER_CONFIGS = None
cfg.TRAIN.SGLORA = edict()
cfg.TRAIN.SGLORA.ENABLE = False
cfg.TRAIN.SGLORA.LAYER_CONFIGS = None
cfg.TRAIN.SGLORA.ENTROPY_LAM_MAX = 1e-4
cfg.TRAIN.SGLORA.WARMUP_RATIO = 0.33
cfg.TRAIN.SGLORA.ANNEAL_RATIO = 0.33
cfg.TRAIN.SGLORA.GROUP_LASSO_LAM_MAX = 1e-5
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# UAV-WSP (Weighted Spectral Projection): anti-UAV small-target adapter
cfg.TRAIN.UAV_WSP = edict()
cfg.TRAIN.UAV_WSP.ENABLE = False
cfg.TRAIN.UAV_WSP.LAYER_CONFIGS = None  # None = use WSP_DEFAULT_PRIOR_CONFIG
cfg.TRAIN.UAV_WSP.ENTROPY_LAM_MAX = 1e-4
cfg.TRAIN.UAV_WSP.WARMUP_RATIO = 0.33
cfg.TRAIN.UAV_WSP.ANNEAL_RATIO = 0.33
cfg.TRAIN.UAV_WSP.GROUP_LASSO_LAM_MAX = 1e-5
cfg.TRAIN.UAV_WSP.FOCUS_LAM_MAX = 1e-5
cfg.TRAIN.UAV_WSP.SPECTRAL_BETA = 1.0
cfg.TRAIN.SAVE_BEST = True               # save best model based on validation metric
cfg.TRAIN.SAVE_BEST_METRIC = "Loss/total"  # metric key to track (lower is better)
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.SAVE_EPOCHS = []  # extra epochs to save checkpoints (e.g. [20])
cfg.TRAIN.AMP = False
# Temporal Smoothness: anti-UAV motion continuity prior
cfg.TRAIN.TEMPORAL_SMOOTHNESS = edict()
cfg.TRAIN.TEMPORAL_SMOOTHNESS.ENABLE = False
cfg.TRAIN.TEMPORAL_SMOOTHNESS.LOSS_WEIGHT = 0.05

cfg.TRAIN.CE_START_EPOCH = 20  # candidate elimination start epoch
cfg.TRAIN.CE_WARM_EPOCH = 80  # candidate elimination warm up epoch
cfg.TRAIN.DROP_PATH_RATE = 0.1  # drop path rate for ViT backbone

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500
cfg.TEST.CHECKPOINT = None

# TEST.KALMAN_FILTER 鈥?adaptive Kalman + adaptive search factor (inference only)
cfg.TEST.KALMAN_FILTER = edict()
cfg.TEST.KALMAN_FILTER.ENABLE = False
cfg.TEST.KALMAN_FILTER.DT = 1.0
cfg.TEST.KALMAN_FILTER.BASE_PROCESS_NOISE = 0.01
cfg.TEST.KALMAN_FILTER.VELOCITY_NOISE_SCALE = 10.0
cfg.TEST.KALMAN_FILTER.SIZE_NOISE_SCALE = 2.0
cfg.TEST.KALMAN_FILTER.BASE_OBS_NOISE = 5.0
cfg.TEST.KALMAN_FILTER.ADAPT_THRESHOLD = 5.0
cfg.TEST.KALMAN_FILTER.ADAPT_GAIN = 2.0
cfg.TEST.KALMAN_FILTER.DECAY_RATE = 0.95
cfg.TEST.KALMAN_FILTER.MAX_Q_SCALE = 10.0
cfg.TEST.KALMAN_FILTER.CONF_THRESHOLD = 0.3
cfg.TEST.KALMAN_FILTER.MIN_SEARCH_FACTOR = 2.5
cfg.TEST.KALMAN_FILTER.MAX_SEARCH_FACTOR = 5.0
cfg.TEST.KALMAN_FILTER.SAFETY_MARGIN = 1.5
cfg.TEST.KALMAN_FILTER.MAX_LOW_CONF_FRAMES = 5


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, allow_unicode=True)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename, encoding='utf-8') as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)

