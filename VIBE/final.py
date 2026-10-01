# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import os
import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from lib.models.vibe import VIBE_Demo
from lib.utils.renderer import Renderer
from lib.dataset.inference import Inference
from lib.utils.smooth_pose import smooth_pose
from lib.data_utils.kp_utils import convert_kps
from lib.utils.pose_tracker import run_posetracker
from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)

MIN_NUM_FRAMES = 25

def check_full_body_visibility(joints3d):
    """ Check if all keypoints for full-body (head to feet) are present. """
    required_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    if np.all(joints3d[:, required_keypoints, 2] > 0.1):  # Confidence score threshold
        return True
    return False

def check_multiple_persons(tracking_results):
    if len(tracking_results) > 1:
        print(f"Warning: More than one person detected. Skipping this video.")
        return True
    return False

def handle_omitted_video(video_file, omitted_path):
    """ Safely move omitted video to the omitted folder. """
    video_name = os.path.basename(video_file)
    omitted_video_path = os.path.join(omitted_path, video_name)

    if os.path.exists(video_file):
        try:
            shutil.copy(video_file, omitted_video_path)
            print(f"Omitted video successfully copied: {omitted_video_path}")
            os.remove(video_file)
        except Exception as e:
            print(f"❗ Error handling omitted video: {video_file} -> {omitted_path}. Error: {e}")
    else:
        print(f"❗ Video file not found: {video_file} — Skipping move operation.")

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    video_file = args.vid_file

    if not os.path.isfile(video_file):
        print(f'Input video "{video_file}" does not exist!')
        return

    output_path = args.output_folder
    omitted_path = os.path.join(output_path, 'omitted')
    os.makedirs(omitted_path, exist_ok=True)

    image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)
    print(f'Input video number of frames {num_frames}')

    bbox_scale = 1.1
    mot = MPT(
        device=device,
        batch_size=args.tracker_batch_size,
        display=args.display,
        detector_type=args.detector,
        output_format='dict',
        yolo_img_size=args.yolo_img_size,
    )
    tracking_results = mot(image_folder)

    if check_multiple_persons(tracking_results):
        print(f'Omitting video: {video_file}')
        handle_omitted_video(video_file, omitted_path)
        return

    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    model.load_state_dict(ckpt['gen_state_dict'], strict=False)
    model.eval()

    print(f'Running VIBE on each tracklet...')
    vibe_results = {}

    for person_id in tracking_results.keys():
        frames = tracking_results[person_id]['frames']

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=tracking_results[person_id].get('bbox'),
            joints2d=tracking_results[person_id].get('joints2d'),
            scale=bbox_scale,
        )

        dataloader = DataLoader(dataset, batch_size=args.vibe_batch_size, num_workers=16)

        with torch.no_grad():
            pred_joints3d = []

            for batch in dataloader:
                batch = batch.unsqueeze(0).to(device)
                output = model(batch)[-1]
                pred_joints3d.append(output['kp_3d'].reshape(-1, 49, 3))

            pred_joints3d = torch.cat(pred_joints3d, dim=0)

        if not check_full_body_visibility(pred_joints3d.cpu().numpy()):
            print(f'Omitting video (incomplete body detected): {video_file}')
            handle_omitted_video(video_file, omitted_path)
            return

        vibe_results[person_id] = {'joints3d': pred_joints3d.cpu().numpy()}

    pkl_filename = os.path.join(output_path, f"{os.path.basename(video_file).replace('.mp4', '')}_vibe_output.pkl")
    joblib.dump(vibe_results, pkl_filename)
    print(f'Saved VIBE results to {pkl_filename}')

    shutil.copy(video_file, output_path)
    print(f'Saved original video to {output_path}')

    output_video_path = os.path.join(output_path, f"{os.path.basename(video_file).replace('.mp4', '')}_vibe_result.mp4")
    images_to_video(img_folder=image_folder, output_vid_file=output_video_path)
    print(f'Saved VIBE rendered video to {output_video_path}')

    shutil.rmtree(image_folder)
    print('================= END =================')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str, help='input video path or youtube link')
    parser.add_argument('--output_folder', type=str, help='output folder to write results')
    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'])
    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'])
    parser.add_argument('--yolo_img_size', type=int, default=416)
    parser.add_argument('--tracker_batch_size', type=int, default=12)
    parser.add_argument('--staf_dir', type=str, default=None)
    parser.add_argument('--vibe_batch_size', type=int, default=450)
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--run_smplify', action='store_true')
    parser.add_argument('--no_render', action='store_true')
    parser.add_argument('--wireframe', action='store_true')
    parser.add_argument('--sideview', action='store_true')
    parser.add_argument('--save_obj', action='store_true')
    parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--smooth_min_cutoff', type=float, default=0.004)
    parser.add_argument('--smooth_beta', type=float, default=0.7)

    args = parser.parse_args()
    main(args)