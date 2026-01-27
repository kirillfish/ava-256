import argparse
import io
import logging
import os
import sys

from random import sample

import cv2
import skimage
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('tkagg')
import pillow_avif

from PIL import Image
from zipp import Path as ZipPath
from plyfile import PlyData
from tqdm import tqdm

from utils import load_camera_calibration, load_obj
from eyebrow_utils import load_head_pose

AVA_ROOT = "/home/kirillfish/Datasets/ava-256"
SAM_ROOT = '/home/kirillfish/Projects/sam3'
SUBJECT_IDS = ["20230405--1635--AAN112", "20230810--1630--ANX726"]

# Клонируем SAM3 отсюда https://github.com/facebookresearch/sam3
# Чекпойнт sam3.pt берем с HF: https://huggingface.co/facebook/sam3/tree/main

sys.path.append(SAM_ROOT)
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def get_random_frame_ids(base_dir, num=5):
    frame_ids_file = f'{base_dir}/frame_list.csv'
    all_frame_ids = []
    with open(frame_ids_file) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            all_frame_ids.append(line.strip().split(',')[1])
    frame_ids = sample(all_frame_ids, num)
    return frame_ids


def get_verts(base_dir, frame_id):
    path = ZipPath(
        f"{base_dir}/kinematic_tracking/registration_vertices.zip",
        f"{frame_id:06d}.ply",
    )
    verts = np.array([list(v) for v in PlyData.read(io.BytesIO(path.read_bytes()))["vertex"].data])

    # pose to world
    head_pose = load_head_pose(base_dir, frame_id)
    verts_h = np.hstack([verts, np.ones((verts.shape[0], 1))])
    verts_world = (head_pose @ verts_h.T).T
    return verts_world


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='./eyebrow_visualization')
    parser.add_argument('--camera-id-subsample', default=20, type=int,
                        help="how many random camera_ids we take for every subject_id")
    parser.add_argument('--frame-id-subsample', default=30, type=int,
                        help="how many random frame_ids we take for every subject_id")
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    args = parse_args()
    OUT_DIR = args.out_dir
    FRAME_ID_SUBSAMPLE = args.frame_id_subsample
    CAMERA_ID_SUBSAMPLE = args.camera_id_subsample

    model = build_sam3_image_model(
        bpe_path=os.path.join(SAM_ROOT, 'sam3/assets/bpe_simple_vocab_16e6.txt.gz'),
        checkpoint_path=os.path.join(SAM_ROOT, 'sam3.pt'),
        load_from_HF=False,
        enable_inst_interactivity=False,
    )

    processor = Sam3Processor(model, confidence_threshold=0.5)

    os.makedirs(OUT_DIR, exist_ok=True)
    mesh = load_obj("assets/face_topology.obj")
    opening_kernel = np.ones((3, 3), np.uint8)

    for subject_id in SUBJECT_IDS:
        logging.info(f'processing subject {subject_id}')
        base_dir = f"{AVA_ROOT}/{subject_id}/decoder/"
        frame_ids_subsample = get_random_frame_ids(base_dir, num=FRAME_ID_SUBSAMPLE)
        all_camera_params = load_camera_calibration(f"{base_dir}/camera_calibration.json")
        camera_ids_subsample = sample(all_camera_params.keys(), CAMERA_ID_SUBSAMPLE)
        for camera_id in camera_ids_subsample:
            logging.info(f'processing camera {camera_id}')
            for frame_id in tqdm(frame_ids_subsample):
                path = ZipPath(
                                base_dir + "image/" + f"cam{camera_id}.zip",
                                f"cam{camera_id}/{int(frame_id):06d}.avif",
                            )
                try:
                    img_bytes = path.read_bytes()
                except FileNotFoundError:
                    logging.error(f"no file {path.name}")
                    continue
                image = Image.open(io.BytesIO(img_bytes))

                inference_state = processor.set_image(image)
                processor.reset_all_prompts(inference_state)
                inference_state = processor.set_text_prompt(state=inference_state, prompt="eyebrow")
                if inference_state['masks'].shape[0] > 0:
                    masks_2d_direct = inference_state['masks'].cpu().numpy()[0, 0, :, :]
                    opening = cv2.morphologyEx(masks_2d_direct.astype('uint8'), cv2.MORPH_OPEN, opening_kernel)
                    area_filtered = skimage.morphology.area_opening(opening, area_threshold=150, connectivity=2)
                else:
                    print(inference_state['masks'].shape)
                    area_filtered = np.zeros_like(image) #([1, 1, 1024, 667])

                plt.figure(figsize=(12, 20))
                plt.imshow(np.array(image))
                plt.imshow(area_filtered, alpha=0.3)
                plt.savefig(f"{OUT_DIR}/{subject_id}_{camera_id}_{frame_id}.jpg")
                plt.close()
