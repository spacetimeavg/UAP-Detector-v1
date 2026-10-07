import cv2
import time
import os
from ultralytics import YOLO

RUNNER_ID = os.getenv("RUNNER_ID", "1")
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

model = YOLO("yolov8n.pt")
KNOWN_IGNORE_CLASSES = [4, 14]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Przykładowa lista Twoich kamer (użyj swoich źródeł)
CAMERAS = {
    "krakow_market": "URL_LUB_RTSP_STREAM_1",
    "cam_2": "URL_LUB_RTSP_STREAM_2",
    "cam_3": "URL_LUB_RTSP_STREAM_3"
}

def get_camera_stream(url):
    """Próba nawiązania stabilnego połączenia ze streamem"""
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def run_monitoring_session(duration_seconds=780):
    print(f"=== START MONITORA (Node {RUNNER_ID}) ===")
    start_time = time.time()
    
    # Inicjalizacja połączeń
    caps = {cam_id: get_camera_stream(url) for cam_id, url in CAMERAS.items()}
    previous_frames = {}

    frame_count = 0

    while time.time() - start_time < duration_seconds:
        for cam_id, cap in caps.items():
            if not cap.isOpened():
                print(f"⚠️ [RECONNECT] Ponowne łączenie z kamerą: {cam_id}")
                caps[cam_id] = get_camera_stream(CAMERAS[cam_id])
                continue

            ret, frame = cap.read()
            
            if not ret or frame is None:
                print(f"❌ [BRAK KLATKI] Błąd odczytu z {cam_id}. Resetowanie połączenia...")
                cap.release()
                caps[cam_id] = get_camera_stream(CAMERAS[cam_id])
                continue

            frame_count += 1
            
            # --- ZAPIS TESTOWY PRZY PIERWSZEJ KLATCE ---
            # Daje 100% pewności, że repozytorium utworzy folder i sprawdzi dostęp do kamery
            if frame_count == 1:
                cv2.imwrite(f"{SCREENSHOT_DIR}/test_node{RUNNER_ID}_{cam_id}.jpg", frame)
                print(f"✅ Zapisano klatkę testową połączenia dla {cam_id}")

            # Przetwarzanie detekcji ruchowej i UAP
            process_frame(frame, cam_id, previous_frames)

        time.sleep(0.5) # Przerwa między próbkowaniem

    # Zwolnienie zasobów
    for cap in caps.values():
        cap.release()
    print(f"=== ZAKOŃCZONO SESJĘ (Przetworzono {frame_count} klatek) ===")

def process_frame(frame, cam_id, previous_frames):
    height, width, _ = frame.shape
    sky_cutoff = int(height * 0.35) if "krakow" in str(cam_id).lower() else int(height * 0.70)
    sky_roi = frame[0:sky_cutoff, 0:width]

    resized = cv2.resize(sky_roi, (640, int(360 * (sky_cutoff / height))))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    if cam_id not in previous_frames:
        previous_frames[cam_id] = blurred
        return

    frame_delta = cv2.absdiff(previous_frames[cam_id], blurred)
    previous_frames[cam_id] = blurred

    _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{SCREENSHOT_DIR}/node{RUNNER_ID}_{cam_id}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            # YOLO Verification
            results = model(filename, verbose=False)
            # Logika zapisywania pliku na dysku
            break

if __name__ == "__main__":
    run_monitoring_session(duration_seconds=780)
