import cv2
import time
import os
from ultralytics import YOLO
# Tu jest Twój istniejący moduł do Google Sheets (np. fetch_cameras)
# from sheets_config import get_cameras_from_sheet 

RUNNER_ID = os.getenv("RUNNER_ID", "1")
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

model = YOLO("yolov8n.pt")
KNOWN_IGNORE_CLASSES = [4, 14]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def get_camera_stream(url):
    """Próba połączenia ze strumieniem"""
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def run_monitoring_session(duration_seconds=780):
    print(f"=== START MONITORA (Node {RUNNER_ID}) ===")
    
    # Pobieramy słownik/listę kamer bezpośrednio z Google Sheets!
    # Format oczekiwany: {"id_kamery": "URL_STREAMU", ...}
    cameras_from_sheet = get_cameras_from_sheet() 
    print(f"[INFO] Załadowano {len(cameras_from_sheet)} źródeł z arkusza.")

    start_time = time.time()
    caps = {}
    retry_counts = {}

    # Nawiązujemy pierwsze połączenia
    for cam_id, url in cameras_from_sheet.items():
        caps[cam_id] = get_camera_stream(url)
        retry_counts[cam_id] = 0

    previous_frames = {}
    frame_count = 0

    while time.time() - start_time < duration_seconds:
        for cam_id, url in cameras_from_sheet.items():
            cap = caps.get(cam_id)

            # Jeśli strumień nie działa, próbujemy go zrestartować MAX 3 RAZY
            if cap is None or not cap.isOpened():
                if retry_counts[cam_id] < 3:
                    retry_counts[cam_id] += 1
                    print(f"⚠️ [RECONNECT {retry_counts[cam_id]}/3] Kamera: {cam_id}")
                    caps[cam_id] = get_camera_stream(url)
                continue

            ret, frame = cap.read()
            
            if not ret or frame is None:
                if retry_counts[cam_id] < 3:
                    retry_counts[cam_id] += 1
                    print(f"❌ [BRAK KLATKI] Błąd odczytu z {cam_id}. Ponawianie ({retry_counts[cam_id]}/3)...")
                    cap.release()
                    caps[cam_id] = get_camera_stream(url)
                continue

            # Resetujemy licznik błędów, skoro klatka przyszła poprawnie
            retry_counts[cam_id] = 0
            frame_count += 1

            # Klatka testowa na sam start
            if frame_count == 1:
                cv2.imwrite(f"{SCREENSHOT_DIR}/test_node{RUNNER_ID}_{cam_id}.jpg", frame)
                print(f"✅ Zapisano klatkę testową dla {cam_id}")

            # Wywołujemy naszą detekcję z maską ROI i filtrem owadów
            process_frame(frame, cam_id, previous_frames)

        time.sleep(0.5)

    for cap in caps.values():
        if cap:
            cap.release()
    print(f"=== ZAKOŃCZONO SESJĘ (Przetworzono klatek: {frame_count}) ===")
