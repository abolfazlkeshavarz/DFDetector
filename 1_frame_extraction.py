#GPU was 4090

import os
import cv2
import multiprocessing as mp

def extract_all_frames(video_path_output_dir):
    video_path, output_dir = video_path_output_dir
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    try:
        reader = cv2.cudacodec.createVideoReader(video_path)
        use_gpu = True
        print(f"[GPU] {video_name}")
    except cv2.error:
        print(f"[CPU] {video_name}")
        reader = cv2.VideoCapture(video_path)
        use_gpu = False

    frame_number = 0
    saved_count = 0

    while True:
        if use_gpu:
            success, gpu_frame = reader.nextFrame()
            if not success:
                break
            frame = gpu_frame.download()
        else:
            ret, frame = reader.read()
            if not ret:
                break

        frame_filename = f"{video_name}_frame_{frame_number:05d}.jpg"
        frame_path = os.path.join(video_output_dir, frame_filename)
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_number += 1
        saved_count += 1

    if not use_gpu:
        reader.release()

    print(f"✅ [{video_name}] Saved {saved_count} frames.")


def extract_all_videos_parallel(input_dir, output_dir, num_workers=55):
    video_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".mp4")
    ]

    tasks = [(vf, output_dir) for vf in video_files]

    print(f"🧵 Starting parallel extraction with {num_workers} workers...")
    with mp.get_context("spawn").Pool(processes=num_workers) as pool:
        pool.map(extract_all_frames, tasks)


if __name__ == "__main__":
    print("CUDA-enabled device count:", cv2.cuda.getCudaEnabledDeviceCount())
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        cv2.cuda.printCudaDeviceInfo(0)
    input_dir = "/home/hatef/Documents/DFPlatter/C0/C0/Train/fake"
    output_dir = "/home/hatef/Documents/DFPlatter/C0/C0/Train/fake"
    extract_all_videos_parallel(input_dir, output_dir, num_workers=55)
