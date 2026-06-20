from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.ostrack.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/ostrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    # Allow YAML to specify TEST.CHECKPOINT:
    # - If set to 'best', use OSTRack_best.pth.tar
    # - If set to an absolute or relative path, use it directly
    # - Otherwise fallback to epoch-based naming using TEST.EPOCH
    checkpoint_cfg = getattr(cfg.TEST, 'CHECKPOINT', None)
    if checkpoint_cfg is not None:
        if isinstance(checkpoint_cfg, str) and checkpoint_cfg.lower() == 'best':
            params.checkpoint = os.path.join(save_dir, "checkpoints/train/ostrack/%s/OSTrack_best.pth.tar" % yaml_name)
        else:
            # If user provided a path, allow absolute or relative to checkpoints dir
            if os.path.isabs(checkpoint_cfg):
                params.checkpoint = checkpoint_cfg
            else:
                params.checkpoint = os.path.join(save_dir, "checkpoints/train/ostrack/%s/%s" % (yaml_name, checkpoint_cfg))
    else:
        params.checkpoint = os.path.join(save_dir, "checkpoints/train/ostrack/%s/OSTrack_ep%04d.pth.tar" %
                                         (yaml_name, cfg.TEST.EPOCH))

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
