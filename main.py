import cv2
import numpy as np
import torch
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

try:
    from ocsort.ocsort import OCSort
except ImportError:
    print("Error: Library 'ocsort' tidak ditemukan. Pastikan sudah diinstal atau file ocsort.py berada di PATH.")
    print("Coba jalankan: !pip install git+https://github.com/ocsort/ocsort.git")
    exit() # Hentikan eksekusi jika ocsort tidak ditemukan

from collections import deque
from datetime import datetime, timedelta

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 with OC-SORT Video File Tracking (with enhanced visualization)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input argument - hanya mendukung file video
    parser.add_argument('--video', type=str, required=True,
                        help='Path to video file for tracking')

    # YOLO arguments
    parser.add_argument('--weights', type=str, default='yolov8n.pt',
                        help='Path to YOLOv8 weights file (e.g., best.pt or yolov8n.pt)')

    parser.add_argument('--conf', type=float, default=0.3, # BISA DIUBAH: Coba 0.1 atau 0.05 jika masih tidak terdeteksi
                        help='Confidence threshold for detection')

    parser.add_argument('--iou', type=float, default=0.3,
                        help='IoU threshold for NMS')

    # Output arguments
    parser.add_argument('--output', type=str,
                        default='results/output_video.mp4',
                        help='Path to save output video')

    parser.add_argument('--save-video', action='store_true',
                        help='Save processed video to file')

    parser.add_argument('--show-live', action='store_true', default=False,
                        help='Show live processing window (only works in local environments)')

    # OC-SORT specific parameters
    parser.add_argument('--delta-t', type=int, default=3,
                        help='Time window parameter for OC-SORT')
    parser.add_argument('--inertia', type=float, default=0.2,
                        help='Motion inertia parameter for OC-SORT')
    
    # New argument for display width to control zoom
    parser.add_argument('--max-display-width', type=int, default=1280,
                        help='Maximum width for the display window (rescales if original is larger)')

    # New argument for fixed start time string
    parser.add_argument('--fixed-start-time', type=str, default="15:12:01",
                        help='Fixed start time string for display (HH:MM:SS)')

    return parser.parse_args()

# Inisialisasi dictionary untuk warna unik per ID
colors = {}
np.random.seed(42) # Agar warna selalu sama setiap dijalankan
def get_color(id):
    if id not in colors:
        colors[id] = tuple(map(int, np.random.randint(0, 255, size=3).tolist()))
    return colors[id]

def process_stream(source, model, tracker, class_names, args):
    """Process video stream (file only) with OC-SORT tracking and enhanced visualization"""

    print(f"\n🔍 Processing video file: {source}")
    cap_source = source

    print("=====================================")

    # Buka video source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source at {cap_source}")
        return

    # Ambil properti video asli
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print("⚠️ Warning: Could not determine total frames. Processing until end of file or 'q' press.")
        total_frames_display = None # Untuk tqdm
    else:
        total_frames_display = total_frames
        
    print(f"Dimensi frame asli: {frame_width}x{frame_height} at {fps:.2f} FPS")

    # Hitung faktor skala untuk resize frame tampilan
    scale_factor = min(1.0, args.max_display_width / frame_width)
    print(f"Faktor skala tampilan: {scale_factor:.2f}")

    # Setup output video writer jika saving diaktifkan
    video_writer = None
    if args.save_video:
        output_video_path = Path(args.output)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec untuk MP4
        video_writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            fps if fps > 0 else 25.0,  # Default to 25 FPS jika tidak diketahui
            (frame_width, frame_height) # Simpan dengan resolusi asli
        )
        print(f"💾 Output video will be saved to: {output_video_path}")

    # Variabel untuk jejak pergerakan kendaraan (diperlukan untuk menggambar garis jejak)
    tracks_history = {} # Dictionary untuk menyimpan deque per ID

    # Penghitung kendaraan dan ID yang sudah dihitung
    count = 0
    counted_ids = set()

    # Garis horizontal untuk counting
    # Sesuaikan koordinat ini agar sesuai dengan resolusi video Anda
    # Misal, 75% dari tinggi frame untuk y, dan 29-72% untuk lebar X
    line_y = int(frame_height * 0.70)
    line_left = int(frame_width * 0.19)
    line_right = int(frame_width * 0.85)

    # Variabel waktu dan FPS
    processing_start_time = time.time() # Waktu nyata script dimulai
    frame_process_times = deque(maxlen=30) # Simpan waktu 30 frame terakhir untuk FPS rata-rata

    # --- Waktu Mulai Yang Disesuaikan (menggunakan datetime) ---
    today = datetime.now().date()
    fixed_start_hour, fixed_start_minute, fixed_start_second = map(int, args.fixed_start_time.split(':'))
    
    fixed_start_datetime_obj = datetime(today.year, today.month, today.day,
                                        fixed_start_hour, fixed_start_minute, fixed_start_second)
    # --- End Waktu Mulai Yang Disesuaikan ---

    print("🎬 Starting processing... Press 'q' to quit (if 'show-live' is true)")

    try:
        with tqdm(total=total_frames_display, desc="Processing Video", unit="frame") as pbar:
            while True:
                frame_read_start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("\n✅ End of video file reached or error reading frame.")
                    break

                # Deteksi objek menggunakan YOLO
                model_inference_start_time = time.time()
                # Pastikan model berjalan di GPU jika tersedia
                results = model(frame, conf=args.conf, iou=args.iou, verbose=False, device=model.device)[0]
                model_inference_end_time = time.time()

                detections = []
                for r in results.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, score, class_id = r
                    # Model kustom  dengan 1 kelas akan memiliki class_id = 0
                    if int(class_id) == 0: # untuk car
                        detections.append([x1, y1, x2, y2, score, class_id])
                
                # Convert to torch tensor
                if len(detections) > 0:
                    detections_tensor = torch.tensor(detections)
                else:
                    detections_tensor = torch.zeros((0, 7))

                # Update tracker dengan deteksi
                # OCSort biasanya membutuhkan format [x1, y1, x2, y2, score, class_id]
                ocsort_tracks = tracker.update(detections_tensor, None)

                # Reset aktif ID untuk frame ini
                active_track_ids_in_frame = set()

                for track in ocsort_tracks:
                    # Pastikan urutan output OC-SORT: x1, y1, x2, y2, track_id, cls, conf
                    x1, y1, x2, y2, track_id, cls_from_track, conf_from_track = track
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    track_id = int(track_id)
                    class_id = int(cls_from_track) # Ambil class_id dari output tracker

                    active_track_ids_in_frame.add(track_id)

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    color = get_color(track_id)

                    # Simpan jejak pergerakan (center point)
                    if track_id not in tracks_history:
                        tracks_history[track_id] = deque(maxlen=30)
                    tracks_history[track_id].append((cx, cy))

                    # Gambar kotak dan ID
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    class_name = class_names.get(class_id, f"Unknown-{class_id}") 
                    text_label = f"ID: {track_id} | {class_name} {conf_from_track:.2f}"
                    
                    (text_width, text_height), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1), color, -1)
                    cv2.putText(frame, text_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Deteksi melewati garis dari atas ke bawah
                    if cy >= line_y and cx > line_left and cx < line_right and track_id not in counted_ids:
                        count += 1
                        counted_ids.add(track_id)
                
                # Hapus jejak yang tidak lagi aktif
                keys_to_delete = [id for id in tracks_history if id not in active_track_ids_in_frame]
                for id in keys_to_delete:
                    if id in tracks_history: # Pastikan kunci masih ada sebelum menghapus
                        del tracks_history[id]

                # Gambar garis pembatas (horizontal)
                if line_y >= 0 and line_y < frame_height:
                    cv2.line(frame, (line_left, line_y), (line_right, line_y), (0, 0, 255), 3)

                # Gambar jejak pergerakan
                for track_id, points in tracks_history.items():
                    color = get_color(track_id)
                    for i in range(1, len(points)):
                        prev_point = points[i - 1]
                        curr_point = points[i]
                        # Pastikan poin-poin berada dalam batas frame sebelum menggambar garis
                        if (0 <= prev_point[0] < frame_width and 0 <= prev_point[1] < frame_height and
                            0 <= curr_point[0] < frame_width and 0 <= curr_point[1] < frame_height):
                            cv2.line(frame, prev_point, curr_point, color, 2)

                # --- Informasi Overlay Kanan Atas ---
                current_elapsed_time = time.time() - processing_start_time
                
                current_display_datetime = fixed_start_datetime_obj + timedelta(seconds=current_elapsed_time)
                current_display_time_str = current_display_datetime.strftime("%H:%M:%S")

                text_count = f"Total Kendaraan (Mobil): {count}"
                text_time = f"Waktu: {current_display_time_str}"

                text_size_count, _ = cv2.getTextSize(text_count, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                text_size_time, _ = cv2.getTextSize(text_time, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

                max_text_width = max(text_size_count[0], text_size_time[0])

                padding = 15
                # Koordinat latar belakang teks untuk count
                bg_rect_top_left_count = (frame_width - max_text_width - padding * 2, padding)
                bg_rect_bottom_right_count = (frame_width - padding, padding + text_size_count[1] + padding)
                
                # Koordinat latar belakang teks untuk waktu
                bg_rect_top_left_time = (frame_width - max_text_width - padding * 2, bg_rect_bottom_right_count[1] + padding)
                bg_rect_bottom_right_time = (frame_width - padding, bg_rect_top_left_time[1] + text_size_time[1] + padding)

                # Gambar background transparan untuk teks
                overlay = frame.copy()
                cv2.rectangle(overlay, bg_rect_top_left_count, bg_rect_bottom_right_count, (0, 0, 0), -1)
                cv2.rectangle(overlay, bg_rect_top_left_time, bg_rect_bottom_right_time, (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame) # Gabungkan overlay dengan frame

                # Tulis teks
                cv2.putText(frame, text_count, (bg_rect_top_left_count[0] + padding, bg_rect_top_left_count[1] + text_size_count[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, text_time, (bg_rect_top_left_time[0] + padding, bg_rect_top_left_time[1] + text_size_time[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                # --- Akhir Informasi Overlay ---

                # Save frame jika video writer diaktifkan
                if video_writer:
                    video_writer.write(frame)

                # Hitung FPS real-time untuk tampilan
                frame_end_time = time.time()
                frame_process_times.append(frame_end_time - frame_read_start_time)
                avg_frame_time = sum(frame_process_times) / len(frame_process_times)
                current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

                # Resize frame untuk tampilan live (jika diperlukan)
                display_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

                # Tampilkan FPS di kiri atas window tampilan
                cv2.putText(display_frame, f"FPS: {current_fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Show live preview jika diaktifkan
                if args.show_live:
                    try:
                        cv2.imshow("Tracking Kendaraan (YOLOv8 + OC-SORT)", display_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("\n⏹️  Stopping by user request...")
                            break
                    except cv2.error as e:
                        print(f"⚠️ Warning: Could not display live window. Make sure you're in a desktop environment. Error: {e}")
                        args.show_live = False # Matikan fitur ini jika error

                pbar.update(1)

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")

    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        if args.show_live:
            cv2.destroyAllWindows()

        final_elapsed_time = time.time() - processing_start_time
        if pbar.n > 0:
            print(f"\n✅ Processing completed!")
            print(f"  • Processed {pbar.n} frames in {final_elapsed_time:.2f} seconds")
            if final_elapsed_time > 0:
                print(f"  • Overall Average FPS: {pbar.n / final_elapsed_time:.2f}")

            hours, remainder = divmod(final_elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # --- Format Waktu Output Akhir ---
            start_time_formatted_final = fixed_start_datetime_obj.strftime("%H:%M:%S")
            end_time_datetime_final = fixed_start_datetime_obj + timedelta(seconds=final_elapsed_time)
            end_time_formatted_final = end_time_datetime_final.strftime("%H:%M:%S")

            summary = f"""
Waktu testing: {start_time_formatted_final} - {end_time_formatted_final}
Total waktu: {int(hours)} jam {int(minutes)} menit {int(seconds)} detik
Total Mobil pada jam {start_time_formatted_final} hingga {end_time_formatted_final}: {count}
"""
            print(summary)

            if args.save_video and video_writer:
                print(f"  • Video saved to: {args.output}")

def main():
    args = parse_args()
    
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        device = 'cuda'
        print("Menggunakan GPU:", torch.cuda.get_device_name(0))
    else:
        device = 'cpu'
        print("⚠️ GPU tidak tersedia, menggunakan CPU")

    try:
        print(f"\n🚀 Loading YOLOv8 model from {args.weights}...")
        model = YOLO(args.weights)
        model.to(device)
        class_names = model.names # Dapatkan mapping class_id ke nama

        print("Initializing OC-SORT tracker...")
        tracker = OCSort(
            det_thresh=args.conf,
            iou_threshold=args.iou,
            delta_t=args.delta_t,
            inertia=args.inertia
        )

        source = args.video
        process_stream(source, model, tracker, class_names, args)

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    print("\n✅ YOLOv8 + OC-SORT Selesai!")

if __name__ == "__main__":
    main()