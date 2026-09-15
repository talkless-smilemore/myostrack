# coding=utf-8
import os
import tqdm
import numpy as np
import cv2


def getdir(path):
    os.makedirs(path, exist_ok=True)


def load_text_numpy(path, delimiter=' ', dtype=np.float32):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype)
                return ground_truth_rect
            except:
                pass
        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
        return ground_truth_rect


def modality_suffix(modality):
    """Convert dataset modality names to the suffix used by result files."""
    suffixes = {'visible': 'RGB', 'infrared': 'IR'}
    if modality not in suffixes:
        raise ValueError(f"Unsupported modality: {modality}")
    return suffixes[modality]


def draw_rectangle(img_root, gt_root, track_roots, sequence_name, save_dir, modality='visible'):
    """
    Draw boxes for an Anti-UAV sequence.

    Expected dataset layout:
        img_root/<sequence_name>/infrared.mp4
        img_root/<sequence_name>/visible.mp4
        gt_root/<sequence_name>_IR.txt
        gt_root/<sequence_name>_RGB.txt
    """
    suffix = modality_suffix(modality)
    anno_path = os.path.join(gt_root, f'{sequence_name}_{suffix}.txt')
    ground_truth_rect = load_text_numpy(str(anno_path), delimiter=[',', ' ', '\t'], dtype=np.float64)
    ground_truth_rect = np.atleast_2d(ground_truth_rect)
    
    # Read all algorithm results
    tracking_results = {}
    for algo_name, algo_path in track_roots.items():
        txt_path = os.path.join(algo_path, f'{sequence_name}_{suffix}.txt')
        try:
            tracking_results[algo_name] = load_text_numpy(str(txt_path), delimiter=[',', ' ', '\t'], dtype=np.float64)
            tracking_results[algo_name] = np.atleast_2d(tracking_results[algo_name])
            print(f"Successfully loaded {algo_name}: {txt_path}")
        except Exception as e:
            print(f"Failed to load {algo_name} from {txt_path}: {e}")
            tracking_results[algo_name] = None

    # Color mapping (BGR format)
    color_map = {
        'SACT': (0, 0, 255),          # Red
        'FocusTrack': (255, 0, 0),    # Blue
        'MCITrack': (0, 255, 255),    # Yellow
        'ODTrack': (255, 0, 255),     # Purple
        'DropTrack': (255, 255, 0),   # Cyan
        'GroundTruth': (0, 255, 0),   # Green
    }

    video_path = os.path.join(img_root, sequence_name, f'{modality}.mp4')
    if not os.path.isfile(video_path):
        print(f"Warning: video does not exist: {video_path}")
        return

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Warning: Cannot open video: {video_path}")
        return

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = min(total_frames, len(ground_truth_rect)) if total_frames > 0 else len(ground_truth_rect)

    save_modality_dir = os.path.join(save_dir, sequence_name, modality)
    getdir(save_modality_dir)

    for i in tqdm.tqdm(range(frame_count), desc=f'{sequence_name}_{suffix}'):
        ok, image = video.read()
        if not ok:
            print(f"Warning: Cannot read frame {i + 1} from {video_path}")
            break

        # VideoCapture returns BGR. Convert grayscale thermal frames for colored boxes.
        if len(image.shape) == 2:
            image_draw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_draw = image.copy()
        
        # Draw Ground Truth
        x1, y1, w, h = ground_truth_rect[i, :4]
        if not np.any(np.isnan([x1, y1, w, h])):
            cv2.rectangle(image_draw, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                          color=color_map['GroundTruth'], thickness=2)
        
        # Draw SACT last so its red box remains visible when boxes overlap.
        draw_order = [name for name in tracking_results if name != 'SACT']
        if 'SACT' in tracking_results:
            draw_order.append('SACT')

        # Draw all algorithm results
        for algo_name in draw_order:
            rects = tracking_results[algo_name]
            if rects is not None and i < len(rects):
                x1_algo, y1_algo, w_algo, h_algo = rects[i, :4]
                if not (np.isnan(x1_algo) or np.isnan(y1_algo) or np.isnan(w_algo) or np.isnan(h_algo)):
                    cv2.rectangle(image_draw, 
                                  (int(x1_algo), int(y1_algo)), 
                                  (int(x1_algo + w_algo), int(y1_algo + h_algo)),
                                  color=color_map.get(algo_name, (255, 255, 255)), thickness=2)
        
        # Save image
        save_path = os.path.join(save_modality_dir, "%04d.jpg" % (i + 1))
        cv2.imwrite(save_path, image_draw)

    video.release()


if __name__ == '__main__':
    img_root = "/home/chenxinyi/data/anti_uav/test/"
    gt_root = "/home/chenxinyi/OSTrack-main/output/sglora/test/tracking_results/groundtruth/"
    save_dir = "/home/chenxinyi/OSTrack-main/output/rectangle/"
    
    track_roots = {
        'SACT': "/home/chenxinyi/OSTrack-main/output/sglora/test/tracking_results/ostrack/vitb_384_mae_ce_32x4_ep300_uav_wsp_best384/",
        'FocusTrack': "/home/chenxinyi/OSTrack-main/output/sglora/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300_uav_lora/",
        'MCITrack': "/home/chenxinyi/OSTrack-main/output/sglora/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300_uav_adalora/",
        'ODTrack': "/home/chenxinyi/OSTrack-main/output/sglora/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300_uav_asc_lora_right/",
        'DropTrack': "/home/chenxinyi/OSTrack-main/output/sglora/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300_uav_milora/",
    }
    
    sequence_name_list = ['20190925_134301_1_9']
    modalities = ['infrared']
    
    for sequence in sequence_name_list:
        for modality in modalities:
            try:
                print(f"\nProcessing sequence: {sequence} - Modality: {modality}")
                draw_rectangle(img_root, gt_root, track_roots, sequence, save_dir, modality)
                print(f"Finished processing {sequence} - {modality}")
            except Exception as e:
                print(f"Error processing {sequence} - {modality}: {e}")
