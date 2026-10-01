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

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

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
import json
from collections import defaultdict

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

MIN_NUM_FRAMES = 15

# merging all tracklets for 1 person
def merge_vibe_tracklets(vibe_results):
    """Merge all person tracklets into a single person."""
    # Initialize merged results
    merged_results = {
        'pose': [],
        'shape': [],
        'camera': [],
        'joints': [],
        'vertices': [],
        'frame_indices': [],
    }
    
    all_frame_indices = []
    
    # Iterate through all tracklets
    for person_id, output_dict in vibe_results.items():
        poses = output_dict['pose']
        shapes = output_dict['betas']  # SMPL shape parameters
        cameras = output_dict['pred_cam']  # Camera parameters
        joints = output_dict['joints3d']  # 3D joints
        vertices = output_dict['verts']  # Body mesh vertices
        frame_indices = output_dict.get('frame_indices', range(len(poses)))

        # Append tracklet data to merged_results
        merged_results['pose'].extend(poses.tolist() if isinstance(poses, np.ndarray) else poses)
        merged_results['shape'].extend(shapes.tolist() if isinstance(shapes, np.ndarray) else shapes)
        merged_results['camera'].extend(cameras.tolist() if isinstance(cameras, np.ndarray) else cameras)
        merged_results['joints'].extend(joints.tolist() if isinstance(joints, np.ndarray) else joints)
        merged_results['vertices'].extend(vertices.tolist() if isinstance(vertices, np.ndarray) else vertices)
        all_frame_indices.extend(frame_indices)
    
    # Remove duplicates and sort frame indices
    unique_frame_indices = sorted(set(all_frame_indices))
    merged_results['frame_indices'] = unique_frame_indices

    return merged_results


def save_merged_results(merged_results, output_file):
    """Save merged results to a JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_ndarray_to_list(data):
        if isinstance(data, dict):
            return {key: convert_ndarray_to_list(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [convert_ndarray_to_list(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
    
    merged_results_serializable = convert_ndarray_to_list(merged_results)
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(merged_results_serializable, f, indent=4)
    print(f"Merged results saved to {output_file}")

# # Example usage in your script:
# def save_merged_results(merged_results):
#     # Convert merged_results to a JSON-serializable format
#     merged_results_serializable = convert_ndarray_to_list(merged_results)

#     # Save the results as a JSON file
#     with open("merged_vibe_results.json", "w") as f:
#         json.dump(merged_results_serializable, f)

#     # # Save dropped frames in a text file
#     # with open("dropped_frames.txt", "w") as f:
#     #     f.writelines(f"{frame}\n" for frame in dropped_frames)


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    video_file = args.vid_file

    # ========= [Optional] download the youtube video ========= #
    if video_file.startswith('https://www.youtube.com'):
        print(f'Donwloading YouTube video \"{video_file}\"')
        video_file = download_youtube_clip(video_file, '/tmp')

        if video_file is None:
            exit('Youtube url is not valid!')

        print(f'YouTube Video has been downloaded to {video_file}...')

    if not os.path.isfile(video_file):
        exit(f'Input video \"{video_file}\" does not exist!')

    output_path = os.path.join(args.output_folder, os.path.basename(video_file).replace('.mp4', ''))
    os.makedirs(output_path, exist_ok=True)

    image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)

    print(f'Input video number of frames {num_frames}')
    orig_height, orig_width = img_shape[:2]

    total_time = time.time()

    # ========= Run tracking ========= #
    bbox_scale = 1.1
    if args.tracking_method == 'pose':
        if not os.path.isabs(video_file):
            video_file = os.path.join(os.getcwd(), video_file)
        tracking_results = run_posetracker(video_file, staf_folder=args.staf_dir, display=args.display)
    else:
        # run multi object tracker
        mot = MPT(
            device=device,
            batch_size=args.tracker_batch_size,
            display=args.display,
            detector_type=args.detector,
            output_format='dict',
            yolo_img_size=args.yolo_img_size,
        )
        tracking_results = mot(image_folder)
        
    # --- ADD THESE PRINT STATEMENTS HERE ---
    print("\n--- Detected Tracklets and their Frame Counts ---")
    for person_id, result in tracking_results.items():
        print(f"  Tracklet ID: {person_id}, Number of frames: {result['frames'].shape[0]}")
    print("-----------------------------------------------\n")
    # --- END ADDITION ---

    for person_id, result in tracking_results.items():
        print(f"Tracklet {person_id}: {result['frames'].shape[0]} frames")

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            print(f"Removing tracklet {person_id} with {tracking_results[person_id]['frames'].shape[0]} frames")
            del tracking_results[person_id]

    # ========= Define VIBE model ========= #
    model = VIBE_Demo(
        seqlen=15,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    # ========= Load pretrained weights ========= #
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file, map_location=torch.device('cpu'))
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')

    # ========= Run VIBE on each person ========= #
    print(f'Running VIBE on each tracklet...')
    vibe_time = time.time()
    vibe_results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None

        if args.tracking_method == 'bbox':
            bboxes = tracking_results[person_id]['bbox']
        elif args.tracking_method == 'pose':
            joints2d = tracking_results[person_id]['joints2d']

        frames = tracking_results[person_id]['frames']

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False

        dataloader = DataLoader(dataset, batch_size=args.vibe_batch_size, num_workers=16)

        with torch.no_grad():

            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d, norm_joints2d = [], [], [], [], [], [], []

            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.unsqueeze(0)
                batch = batch.to(device)

                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]

                pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
                pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))
                pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))
                smpl_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2))


            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            smpl_joints2d = torch.cat(smpl_joints2d, dim=0)
            del batch

        # ========= [Optional] run Temporal SMPLify to refine the results ========= #
        if args.run_smplify and args.tracking_method == 'pose':
            norm_joints2d = np.concatenate(norm_joints2d, axis=0)
            norm_joints2d = convert_kps(norm_joints2d, src='staf', dst='spin')
            norm_joints2d = torch.from_numpy(norm_joints2d).float().to(device)

            # Run Temporal SMPLify
            update, new_opt_vertices, new_opt_cam, new_opt_pose, new_opt_betas, \
            new_opt_joints3d, new_opt_joint_loss, opt_joint_loss = smplify_runner(
                pred_rotmat=pred_pose,
                pred_betas=pred_betas,
                pred_cam=pred_cam,
                j2d=norm_joints2d,
                device=device,
                batch_size=norm_joints2d.shape[0],
                pose2aa=False,
            )

            # update the parameters after refinement
            print(f'Update ratio after Temporal SMPLify: {update.sum()} / {norm_joints2d.shape[0]}')
            pred_verts = pred_verts.cpu()
            pred_cam = pred_cam.cpu()
            pred_pose = pred_pose.cpu()
            pred_betas = pred_betas.cpu()
            pred_joints3d = pred_joints3d.cpu()
            pred_verts[update] = new_opt_vertices[update]
            pred_cam[update] = new_opt_cam[update]
            pred_pose[update] = new_opt_pose[update]
            pred_betas[update] = new_opt_betas[update]
            pred_joints3d[update] = new_opt_joints3d[update]

        elif args.run_smplify and args.tracking_method == 'bbox':
            print('[WARNING] You need to enable pose tracking to run Temporal SMPLify algorithm!')
            print('[WARNING] Continuing without running Temporal SMPLify!..')

        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()
        smpl_joints2d = smpl_joints2d.cpu().numpy()

        # Runs 1 Euro Filter to smooth out the results
        if args.smooth:
            min_cutoff = args.smooth_min_cutoff # 0.004
            beta = args.smooth_beta # 1.5
            print(f'Running smoothing on person {person_id}, min_cutoff: {min_cutoff}, beta: {beta}')
            pred_verts, pred_pose, pred_joints3d = smooth_pose(pred_pose, pred_betas,
                                                               min_cutoff=min_cutoff, beta=beta)

        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        joints2d_img_coord = convert_crop_coords_to_orig_img(
            bbox=bboxes,
            keypoints=smpl_joints2d,
            crop_size=224,
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'joints2d_img_coord': joints2d_img_coord,
            'bboxes': bboxes,
            'frame_ids': frames,
        }

        vibe_results[person_id] = output_dict

    del model

    total_frame_indices = set(range(11077))

    end = time.time()
    fps = num_frames / (end - vibe_time)

    all_frame_indices = []
    for person_id, output_dict in vibe_results.items():
        frame_indices = output_dict.get('frame_ids', range(len(output_dict['pose'])))
        all_frame_indices.extend(frame_indices)
    
    present_frame_indices = set(all_frame_indices)

    # print(f"present_frame_indices: {present_frame_indices}")

    missing_frame_indices = total_frame_indices - present_frame_indices

    missing_frame_indices = sorted(missing_frame_indices)
    missing_frame_indices_array = np.array(missing_frame_indices)
    print(f"missing_frame_indices_array shape: {missing_frame_indices_array.shape}")

    # np.savetxt('missing_frames.txt', missing_frame_indices_array, fmt='%d')

    print(f'VIBE FPS: {fps:.2f}')
    total_time = time.time() - total_time
    print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
    print(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

    # print(f'Saving output results to \"{os.path.join(output_path, "vibe_output.pkl")}\".')

    # joblib.dump(vibe_results, os.path.join(output_path, "vibe_output.pkl"))

    # Dictionary to hold all the results
    results_dict = {}

    print(f"vibe_results type: {type(vibe_results)}")
    # print(f"vibe_results shape: {vibe_results.shape}")

    ## code for saving single person
    # merged_results = merge_vibe_tracklets(vibe_results)

    # print(len(merged_results['pose']))

    ## code for saving multi person
    print("\n=== SAVING MULTI-PERSON RESULTS ===")
    for person_id, output_dict in vibe_results.items():
        person_results = {
            'pose': output_dict['pose'].tolist(),
            'shape': output_dict['betas'].tolist(),
            'camera': output_dict['pred_cam'].tolist(),
            'joints': output_dict['joints3d'].tolist(),
            'vertices': output_dict['verts'].tolist(),
            'frame_indices': output_dict['frame_ids'].tolist()
        }
        
        filename = os.path.join(args.output_folder, f'person_{person_id}_vibe_results.json')
        with open(filename, 'w') as f:
            json.dump(person_results, f, indent=4)
        print(f"Person {person_id} saved with {len(person_results['pose'])} frames")

    # Save merged results
    # save_merged_results(merged_results, 'merged_vibe_results.json')
    # output_path = os.path.join(args.output_folder, 'merged_vibe_results.json')
    # save_merged_results(merged_results, output_path)

    
    print(f"Total frames in video: {num_frames}")

    # start commenting out
    if not args.no_render:
        # ========= Render results as a single video ========= #
        renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=args.wireframe)

        output_img_folder = f'{image_folder}_output'
        os.makedirs(output_img_folder, exist_ok=True)

        print(f'Rendering output video, writing frames to {output_img_folder}')

        # prepare results for rendering
        frame_results = prepare_rendering_results(vibe_results, num_frames)
        mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

        image_file_names = sorted([
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for frame_idx in tqdm(range(len(image_file_names))):
            # img_fname = image_file_names[frame_idx]
            # img = cv2.imread(img_fname)
            img = np.ones((orig_height, orig_width, 3), dtype=np.uint8) * 255

            if args.sideview:
                side_img = np.zeros_like(img)

            for person_id, person_data in frame_results[frame_idx].items():
                frame_verts = person_data['verts']
                frame_cam = person_data['cam']

                mc = mesh_color[person_id]

                mesh_filename = None

                # if args.save_obj:
                #     mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:04d}')
                #     os.makedirs(mesh_folder, exist_ok=True)
                #     mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

                img = renderer.render(
                    img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    mesh_filename=mesh_filename,
                )

                if args.sideview:
                    side_img = renderer.render(
                        side_img,
                        frame_verts,
                        cam=frame_cam,
                        color=mc,
                        angle=270,
                        axis=[0,1,0],
                    )

            if args.sideview:
                img = np.concatenate([img, side_img], axis=1)

            cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

            if args.display:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if args.display:
            cv2.destroyAllWindows()

        # ========= Save rendered video ========= #
        vid_name = os.path.basename(video_file)
        save_name = f'{vid_name.replace(".mp4", "")}_vibe_result.mp4'
        save_name = os.path.join(output_path, save_name)
        print(f'Saving result video to {save_name}')
        images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
        # shutil.rmtree(output_img_folder)

    # shutil.rmtree(image_folder)
    print('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str,
                        help='input video path or youtube link')

    parser.add_argument('--output_folder', type=str,
                        help='output folder to write results')

    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--staf_dir', type=str, default='/home/mkocabas/developments/openposetrack',
                        help='path to directory STAF pose tracking method installed.')

    parser.add_argument('--vibe_batch_size', type=int, default=450,
                        help='batch size of VIBE')

    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')

    parser.add_argument('--run_smplify', action='store_true',
                        help='run smplify for refining the results, you need pose tracking to enable it')

    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    parser.add_argument('--smooth', action='store_true',
                        help='smooth the results to prevent jitter')

    parser.add_argument('--smooth_min_cutoff', type=float, default=0.004,
                        help='one euro filter min cutoff. '
                             'Decreasing the minimum cutoff frequency decreases slow speed jitter')

    parser.add_argument('--smooth_beta', type=float, default=0.7,
                        help='one euro filter beta. '
                             'Increasing the speed coefficient(beta) decreases speed lag.')

    args = parser.parse_args()

    main(args)
