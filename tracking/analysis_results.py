import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
# 与 test.py 的 --dataset_name 一致；评测结果目录在 output/test/tracking_results/ostrack/<parameter_name>/
dataset_name = 'anti_uav_ir'


"""stark"""
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-S50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST101'))
"""TransT"""
# trackers.extend(trackerlist(name='TransT_N2', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N2', result_only=True))
# trackers.extend(trackerlist(name='TransT_N4', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N4', result_only=True))
"""pytracking"""
# trackers.extend(trackerlist('atom', 'default', None, range(0,5), 'ATOM'))
# trackers.extend(trackerlist('dimp', 'dimp18', None, range(0,5), 'DiMP18'))
# trackers.extend(trackerlist('dimp', 'dimp50', None, range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimp', 'prdimp18', None, range(0,5), 'PrDiMP18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', None, range(0,5), 'PrDiMP50'))
"""ostrack"""
# parameter_name 必须与 test.py 第二个参数、以及 tracking_results/ostrack/<这里> 文件夹名完全一致
# trackers.extend(trackerlist(name='ostrack', parameter_name='my_uav_finetune_99', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack-finetune'))
trackers.extend(trackerlist(
    name='ostrack',
    parameter_name='vitb_384_mae_ce_32x4_ep300_uav_oplora',
    dataset_name=dataset_name,
    run_ids=None,
    display_name='OSTrack-uav-oplora'
))
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300_sf36_tf25', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack256'))


dataset = get_dataset(*[x.strip() for x in dataset_name.split(',')])

# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# 终端打印 AUC / OP50 / OP75 / Precision 等；图保存到 local.py 里 result_plot_path/<第三个参数>/
# plot_results(trackers, dataset, dataset_name, merge_results=True,
#              plot_types=('success', 'norm_prec', 'prec'), force_evaluation=False)
print_results(trackers, dataset, dataset_name, merge_results=True,
              plot_types=('success', 'norm_prec', 'prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
