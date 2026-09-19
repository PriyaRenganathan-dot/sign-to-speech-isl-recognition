# ============================================================
#  DATASET HELPER — Sign Recognition App
#  Final Year Project
#  Downloads or guides setup of ISL / ASL datasets
# ============================================================
#
#  DATASETS USED IN THIS PROJECT:
#
#  1. WLASL (World-Level American Sign Language)
#     - 2000 classes, 21,000+ videos
#     - URL: https://dxli94.github.io/WLASL/
#     - Paper: "Word-level Deep Sign Language Recognition..."
#
#  2. INCLUDE (Indian Sign Language Dataset — IIT Madras)
#     - 263 ISL words, 4287 videos
#     - URL: https://zenodo.org/record/4010759
#     - Best for Tamil Nadu ISL users
#
#  3. LSA64 (Argentina — for transfer learning)
#     - 64 signs, 3200 videos
#     - URL: http://facundoq.github.io/datasets/lsa64/
#
#  HOW TO USE:
#  python download_dataset.py --dataset include
#  python download_dataset.py --dataset wlasl
#  python download_dataset.py --dataset collect  (use your webcam)

import os
import sys
import argparse
import zipfile
import urllib.request
from pathlib import Path

DATASET_DIR = Path('../dataset')

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║           ISL SIGN RECOGNITION — DATASET SETUP          ║
║              Final Year Project — Dataset Helper         ║
╚══════════════════════════════════════════════════════════╝
    """)

def guide_include():
    """INCLUDE Dataset — IIT Madras (Best for Indian Sign Language)"""
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  INCLUDE DATASET (IIT Madras — ISL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✅ RECOMMENDED for your project (Indian Sign Language)

  STEPS:
  1. Go to: https://zenodo.org/record/4010759
  2. Download: INCLUDE.zip (~2.8 GB)
  3. Extract to: ../dataset/include/
  4. Run: python preprocess_include.py

  DATASET INFO:
  • 263 ISL sign classes
  • 4,287 video clips
  • Multiple signers
  • Recorded at IIT Madras

  FOLDER STRUCTURE EXPECTED:
  dataset/
  └── include/
      ├── 001_hello/
      │   ├── signer1_001.mp4
      │   ├── signer2_001.mp4
      │   └── ...
      ├── 002_thank_you/
      └── ...
    """)

def guide_wlasl():
    """WLASL Dataset (American Sign Language — 2000 classes)"""
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WLASL DATASET (American Sign Language)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  This is the dataset from the image you shared (ChatGPT)
  Uses Inception 3D (I3D) model architecture

  STEPS:
  1. Go to: https://dxli94.github.io/WLASL/
  2. Fill Google Form to get download link
  3. Download WLASL_v0.3.json (class list)
  4. Download videos (~14 GB for 2000 classes)
  5. Run: python preprocess_wlasl.py

  DATASET INFO:
  • 2,000 ASL word classes
  • 21,000+ video clips
  • Multiple signers, varied backgrounds

  MODEL PIPELINE (like image shows):
  1. Extract frames from videos (OpenCV)
  2. Load I3D model (pretrained on ImageNet)
  3. Fine-tune dense layers
  4. Output: text label for sign
    """)

def guide_custom():
    """Custom collection using webcam"""
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CUSTOM COLLECTION (Your Own Webcam)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✅ EASIEST to start with for FYP demo!

  STEPS:
  1. Run: python data_collection/collect_data.py
  2. For each of 25 signs:
     - Press SPACE to start
     - Perform the sign 30 times
     - Each recording = 30 frames
  3. Total: 25 signs × 30 sequences × 30 frames
     = 22,500 data points

  RECOMMENDED COLLECTION TIPS:
  • Record in good lighting (face a window)
  • Vary your hand position slightly each time
  • Record both slow and normal speed
  • Have 3-4 different people record if possible
  • Use plain background

  TIME ESTIMATE:
  • 25 signs × 30 sequences × 5 seconds = ~60 minutes
  • Best split into multiple sessions
    """)

def preprocess_video_dataset(video_dir, output_dir, signs_to_use=None):
    """
    Extract MediaPipe keypoints from video files.
    Works for INCLUDE, WLASL, or any video dataset.

    Args:
        video_dir:    Path to videos (organized by class folders)
        output_dir:   Where to save .npy keypoint files
        signs_to_use: List of class names to process (None = all)
    """
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
    except ImportError:
        print("Install: pip install opencv-python mediapipe numpy")
        return

    mp_hands   = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
    )

    SEQUENCE_LEN = 30  # frames to extract per video

    video_dir  = Path(video_dir)
    output_dir = Path(output_dir)

    sign_folders = sorted([d for d in video_dir.iterdir() if d.is_dir()])
    if signs_to_use:
        sign_folders = [d for d in sign_folders if d.name in signs_to_use]

    print(f"Processing {len(sign_folders)} sign classes...")

    for sign_folder in sign_folders:
        sign_name = sign_folder.name
        videos    = list(sign_folder.glob('*.mp4')) + list(sign_folder.glob('*.avi'))

        print(f"  [{sign_name}] — {len(videos)} videos")

        for seq_idx, video_path in enumerate(videos):
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Sample SEQUENCE_LEN evenly-spaced frames
            frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LEN, dtype=int)
            frames_data   = []

            for fi in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if not ret:
                    frames_data.append(np.zeros(126))
                    continue

                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                # Extract keypoints
                lh = np.zeros(63)
                rh = np.zeros(63)
                if results.multi_hand_landmarks:
                    for i, lm in enumerate(results.multi_hand_landmarks[:2]):
                        label = results.multi_handedness[i].classification[0].label
                        kp = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
                        if label == 'Left':
                            lh = kp
                        else:
                            rh = kp
                frames_data.append(np.concatenate([lh, rh]))

            cap.release()

            if len(frames_data) == SEQUENCE_LEN:
                save_dir = output_dir / sign_name / str(seq_idx)
                save_dir.mkdir(parents=True, exist_ok=True)
                for fn, fd in enumerate(frames_data):
                    np.save(str(save_dir / str(fn)), fd)

    hands.close()
    print(f"\n✅ Preprocessing complete → {output_dir}")

def main():
    print_banner()

    parser = argparse.ArgumentParser(description='Dataset Setup Helper')
    parser.add_argument('--dataset', choices=['include', 'wlasl', 'collect', 'preprocess'],
                        default='collect', help='Dataset to set up')
    parser.add_argument('--video-dir',  default='../raw_videos',  help='Input video directory')
    parser.add_argument('--output-dir', default='../dataset',     help='Output directory')
    args = parser.parse_args()

    if args.dataset == 'include':
        guide_include()
    elif args.dataset == 'wlasl':
        guide_wlasl()
    elif args.dataset == 'collect':
        guide_custom()
        print("\n  Run: python data_collection/collect_data.py")
    elif args.dataset == 'preprocess':
        print(f"\nPreprocessing videos from: {args.video_dir}")
        preprocess_video_dataset(args.video_dir, args.output_dir)

if __name__ == '__main__':
    main()
