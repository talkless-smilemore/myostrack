#!/usr/bin/env python3
"""检查跟踪结果与数据集序列是否匹配的辅助脚本。

用法:
  python tools/check_results.py

会打印 dataset 长度、前 30 个序列名、tracker.results_dir、缺失文件数量及示例。
"""
import os
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker


def main():
    dataset_name = 'anti_uav_ir'
    config_name = 'vitb_256_mae_ce_32x4_ep300_uav_neuro_oplora'

    ds = get_dataset(dataset_name)
    print('dataset length =', len(ds))
    print('\nfirst 30 sequence names:')
    for i, s in enumerate(ds[:30]):
        print(i, s.name)

    trk = Tracker('ostrack', config_name, dataset_name)
    print('\ntracker.results_dir =', trk.results_dir)

    # list sample result files
    if os.path.isdir(trk.results_dir):
        files = [f for f in os.listdir(trk.results_dir) if f.lower().endswith('.txt')]
        print('\nfound {} result files (examples):'.format(len(files)))
        for f in files[:20]:
            print('  ', f)
    else:
        print('\nresults dir does not exist:', trk.results_dir)

    missing = []
    for s in ds:
        p = os.path.join(trk.results_dir, s.name + '.txt')
        if not os.path.isfile(p):
            missing.append(s.name)

    print('\nmissing count =', len(missing))
    if missing:
        print('examples missing:', missing[:20])


if __name__ == '__main__':
    main()
