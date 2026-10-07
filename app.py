import cv2
import time
import os
import pandas as pd
from ultralytics import YOLO

# Parametry
RUNNER_ID = os.getenv("RUNNER_ID", "1")
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Prawidłowy adres eksportu CSV z Twojego Arkusza Google
SHEET_URL = "https://docs.google.com/spreadsheets/d/1zGjO7LvDWbewwL5vvmtSL8EFm0wTrfiKniH-a02aTjo/export?format=csv&gid=1919540486"

model = YOLO("yolov8n.pt")
KNOWN_IGNORE_CLASSES = [4, 14]  # 4 = samolot, 14 = ptak
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def fetch_cameras_from_sheet():
    """Pobiera listę kamer bezpośrednio z pliku CSV wygenerowanego z Google Sheets"""
    print(f"[INFO] Pobieranie konfiguracji z Google Sheets: {SHEET_URL}")
    try:
        df = pd.read_csv(SHEET_URL)
        df.columns = df.columns.str.strip()
        
        # Oczekiwane nazwy kolumn: 'id' oraz 'stream_url'
        if 'id' in df.columns and 'stream_url' in df.columns:
            cameras = dict(zip(df['id'], df['stream_url']))
            print(f"✅ [SUCCESS] Załadowano {len(cameras)} kamer z arkusza.")
            return cameras
        else:
            print(f"❌ [BŁĄD STRUKTURY] Brak kolumn 'id' lub 'stream_url'. Znaleziono: {list(df.columns)}")
            return {}
    except Exception as e:
        print(f"❌ [BŁĄD POBIERANIA SHEETS]: {e}")
        return {}

def get_camera_stream(url):
    """Nawiązanie połączenia ze strumieniem OpenCV"""
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def process_frame(frame, cam_id, previous_frames):
    """Maska ROI, filtracja szumów/owadów i weryfikacja przez YOLO"""
    height, width, _ = frame.shape
    
    # Maska ROI: Dla rynku krakowskiego analizujemy tylko niebo (35% górnej części)
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
        area = cv2.contourArea(contour)
        # Odrzucanie owadów/szumów (<50px) oraz wielkich plików przelotowych
        if 50 < area < 3500:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{SCREENSHOT_DIR}/node{RUNNER_ID}_{cam_id}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            # Weryfikacja YOLO
            results = model(filename, verbose=False)
            is_unknown = False
            has_ignored = False

            for r in results:
                if len(r.boxes) == 0:
                    is_unknown = True
                else:
                    for box in r.boxes:
                        if int(box.cls[0]) in KNOWN_IGNORE_CLASSES:
                            has_ignored = True
                        else:
                            is_unknown = True

            # Jeśli to zwykły ptak/samolot, usuwamy wygenerowany plik zrzutu
            if has_ignored and not is_unknown:
                if os.path.exists(filename):
                    os.remove(filename)
            break

def run_monitoring_session(duration_seconds=780):
    print(f"=== START MONITORA (Node {RUNNER_ID}) ===")
    
    cameras_from_sheet = fetch_cameras_from_sheet()
    if not cameras_from_sheet:
        print("⚠️ Brak prawidłowych źródeł kamer. Kończę sesję.")
        return

    start_time = time.time()
    caps = {}
    retry_counts = {cam_id: 0 for cam_id in cameras_from_sheet}

    # Pierwsze nawiązanie połączeń
    for cam_id, url in cameras_from_sheet.items():
        caps[cam_id] = get_camera_stream(url)

    previous_frames = {}
    frame_count = 0
    saved_test_frames = set()

    while time.time() - start_time < duration_seconds:
        for cam_id, url in cameras_from_sheet.items():
            cap = caps.get(cam_id)

            # Limit ponowień połączenia (MAX 3 próby)
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

            # Sukces odczytu -> resetujemy licznik błędów
            retry_counts[cam_id] = 0
            frame_count += 1

            # Klatka podglądowa dla każdej ruszającej kamery
            if cam_id not in saved_test_frames:
                test_filename = f"{SCREENSHOT_DIR}/test_node{RUNNER_ID}_{cam_id}.jpg"
                cv2.imwrite(test_filename, frame)
                saved_test_frames.add(cam_id)
                print(f"✅ Zapisano klatkę podglądową dla {cam_id}")

            process_frame(frame, cam_id, previous_frames)

        time.sleep(0.5)

    for cap in caps.values():
        if cap:
            cap.release()
            
    print(f"=== ZAKOŃCZONO SESJĘ (Przetworzono klatek: {frame_count}) ===")

if __name__ == "__main__":
    run_monitoring_session(duration_seconds=780)
