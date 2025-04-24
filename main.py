import cv2
from collections import deque
from ultralytics import YOLO

# 1️⃣ Load YOLO11-nano model (yolo11n.pt)
model = YOLO('yolo11n.pt')        # Nano model: sub-30 MB, ~30–45 FPS on GPU :contentReference[oaicite:5]{index=5}

# 2️⃣ Sliding window of previous IDs
history = deque(maxlen=30)

# 3️⃣ Real-time detection + BoT-SORT tracking
for result in model.track(
        source=0,                   # Webcam input
        tracker='botsort',          # BoT-SORT built-in :contentReference[oaicite:6]{index=6}
        conf=0.25,                  # Confidence threshold
        iou=0.45,                   # NMS IoU threshold
        imgsz=640,                  # Inference resolution
        stream=True
):
    frame = result.orig_img
    # result.boxes.data: [x1, y1, x2, y2, conf, id]
    tracks = result.boxes.data.cpu().numpy()
    current_ids = {int(t[5]) for t in tracks}

    # 4️⃣ Change-detection logic
    prev_ids = history[-1] if history else set()
    missing = prev_ids - current_ids
    new     = current_ids - prev_ids
    history.append(current_ids)

    # 5️⃣ Draw boxes & IDs
    for x1, y1, x2, y2, conf, tid in tracks:
        x1, y1, x2, y2, tid = map(int, [x1, y1, x2, y2, tid])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # 6️⃣ Overlay missing/new alerts
    y0 = 30
    for mid in missing:
        cv2.putText(frame, f"Missing:{mid}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        y0 += 20
    for nid in new:
        cv2.putText(frame, f"New:{nid}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        y0 += 20

    # 7️⃣ Show
    cv2.imshow('YOLO11 + BoT-SORT', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()