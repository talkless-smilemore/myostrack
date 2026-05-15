from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    otb=DatasetInfo(module=pt % "otb", class_name="OTBDataset", kwargs=dict()),
    nfs=DatasetInfo(module=pt % "nfs", class_name="NFSDataset", kwargs=dict()),
    uav=DatasetInfo(module=pt % "uav", class_name="UAVDataset", kwargs=dict()),
    anti_uav=DatasetInfo(module=pt % "anti_uav", class_name="AntiUAVDataset",
                         kwargs=dict(split='test', modalities='both')),
    anti_uav_ir=DatasetInfo(module=pt % "anti_uav", class_name="AntiUAVDataset",
                            kwargs=dict(split='test', modalities='ir')),
    anti_uav_rgb=DatasetInfo(module=pt % "anti_uav", class_name="AntiUAVDataset",
                             kwargs=dict(split='test', modalities='rgb')),
    # Anti-UAV410 等仅红外、平铺图片时用这个名称即可
    anti_uav410=DatasetInfo(module=pt % "anti_uav", class_name="AntiUAVDataset",
                            kwargs=dict(split='test', modalities='ir', env_attr='anti_uav410_path')),
    # 与 anti_uav410 相同：IR + JSON 约定，适用于 Anti-UAV600 / 410 等（路径见 local.py anti_uav_path）
    anti_uav600=DatasetInfo(module=pt % "anti_uav", class_name="AntiUAVDataset",
                            kwargs=dict(split='test', modalities='ir')),
    anti_uav300=DatasetInfo(module=pt % "anti_uav", class_name="AntiUAVDataset",
                            kwargs=dict(split='test', modalities='both', env_attr='anti_uav300_path')),
    anti_uav300_ir=DatasetInfo(module=pt % "anti_uav", class_name="AntiUAVDataset",
                               kwargs=dict(split='test', modalities='ir', env_attr='anti_uav300_path')),
    anti_uav300_rgb=DatasetInfo(module=pt % "anti_uav", class_name="AntiUAVDataset",
                                kwargs=dict(split='test', modalities='rgb', env_attr='anti_uav300_path')),
    tc128=DatasetInfo(module=pt % "tc128", class_name="TC128Dataset", kwargs=dict()),
    tc128ce=DatasetInfo(module=pt % "tc128ce", class_name="TC128CEDataset", kwargs=dict()),
    trackingnet=DatasetInfo(module=pt % "trackingnet", class_name="TrackingNetDataset", kwargs=dict()),
    got10k_test=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='test')),
    got10k_val=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='val')),
    got10k_ltrval=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='ltrval')),
    lasot=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict()),
    lasot_lmdb=DatasetInfo(module=pt % "lasot_lmdb", class_name="LaSOTlmdbDataset", kwargs=dict()),

    vot18=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict()),
    vot22=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict(year=22)),
    itb=DatasetInfo(module=pt % "itb", class_name="ITBDataset", kwargs=dict()),
    tnl2k=DatasetInfo(module=pt % "tnl2k", class_name="TNL2kDataset", kwargs=dict()),
    lasot_extension_subset=DatasetInfo(module=pt % "lasotextensionsubset", class_name="LaSOTExtensionSubsetDataset",
                                       kwargs=dict()),
)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset
