import json
import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time

DATA_DIR = "./data"

def zero_padding(array):
    threshold = 70
    if array.shape[0] > threshold:
        return array[:threshold, :, :]
    elif array.shape[0] < threshold:
        padding = ((0, threshold - array.shape[0]), (0, 0), (0, 0))
        return np.pad(array, padding, mode='constant', constant_values=0)
    else:
        return array

def extract_landmarks(out_dir: str, workout: str, path: str):
    try:
        rep_frames = json.load(open(f'{path[:-4]}.json'))['images']
    except FileNotFoundError:
        print(f"JSON file not found for {path}. Skipping...")
        return

    mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Could not open video {path}.")
        return
    
    data = []
    rep_frame = {}
    frame_count = 0 

    with mp_pose as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                rep_count = rep_frames[frame_count]['rep_count']
                rep_frame[str(int(rep_count))] = rep_frames[frame_count]["frame_numer"]
                landmarks = [[point.x, point.y, point.z] for point in results.pose_landmarks.landmark]
                data.append(landmarks)
            frame_count += 1

    cap.release()
    np_data = np.array(data)

    s = 0
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in rep_frame:
        frame = rep_frame[i]
        file_path = os.path.join(out_dir, f"{os.path.basename(path)[:-4]}-{workout}-{i}.npy")
        np.save(file_path, zero_padding(np_data[s: s + frame]))
        s = frame

    # for i, rep in enumerate(rep_frame):
    #     frame = rep_frame[rep]
    #     file_path = os.path.join(out_dir, f"{os.path.basename(path)[:-4]}-{workout}-{i}.npy")
    #     np.save(file_path, zero_padding(np_data[s: s + frame]))
    #     s = frame

def process_videos(video_files):
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(extract_landmarks, f'./datasets/{d}', d, os.path.join(root, d, f))
            for root, d, f in video_files
        ]
        for future in futures:
            future.result()

def main():
    video_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for d in dirs:
            dir_path = os.path.join(root, d)
            for f in os.listdir(dir_path):
                if f.endswith(".mp4"):
                    # extract_landmarks( f'./datasets/{d}', d, os.path.join(root, d, f) )
                    video_files.append((root, d, f))
    
    if video_files:
        process_videos(video_files)
    else:
        print("No video files found.")

if __name__ == "__main__":
    main()