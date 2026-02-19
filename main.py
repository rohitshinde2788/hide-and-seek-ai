import cv2
import time
from ultralytics import YOLO
from utils.voice import speak

# LOAD TRAINED MODEL
model = YOLO("runs/detect/train16/weights/best.pt")

cap = cv2.VideoCapture(0)

scan_y = 0
last_state = None
last_voice_time = 0
VOICE_DELAY = 5

#  STABILITY MEMORY 
history_mask = []
history_glass = []

# MAIN LOOP
while True:

    ret, frame = cap.read()
    if not ret:
        break

    #  Resize for smooth performance (NO LAG)
    frame = cv2.resize(frame, (640, 480))

    #  Fast Detection (no track -> smooth)
    results = model(frame, conf=0.4, imgsz=640, verbose=False)

    status_mask = None
    status_glass = None
    distance_status = "FAR"

    #  DETECTION 
    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            cls = int(box.cls[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])

            # Draw only ONE clean box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            # DISTANCE ESTIMATION 
            box_width = x2 - x1

            if box_width > 220:
                distance_status = "NEAR"
            elif box_width > 120:
                distance_status = "MID"
            else:
                distance_status = "FAR"

            # CLASS MAP
            # 0 = glass
            # 1 = mask
            # 2 = no_glass
            # 3 = no_mask

            if cls == 0:
                status_glass = "GLASSES"

            elif cls == 1:
                status_mask = "MASK"

            elif cls == 2:
                status_glass = "NO_GLASSES"

            elif cls == 3:
                status_mask = "NO_MASK"

    #  STABILITY 
    if status_mask:
        history_mask.append(status_mask)

    if status_glass:
        history_glass.append(status_glass)

    history_mask = history_mask[-10:]
    history_glass = history_glass[-10:]

    if history_mask.count("MASK") > 5:
        status_mask = "MASK"
    elif history_mask.count("NO_MASK") > 5:
        status_mask = "NO_MASK"

    if history_glass.count("GLASSES") > 5:
        status_glass = "GLASSES"
    elif history_glass.count("NO_GLASSES") > 5:
        status_glass = "NO_GLASSES"

    # ATM DECISION
    current_state = None
    decision_text = None

    # 👉 DISTANCE CHECK FIRST (NEW)
    if distance_status != "NEAR":
        current_state = "MOVE_CLOSER"
        decision_text = "Please come closer to ATM"

        cv2.putText(frame,"MOVE CLOSER",
                    (30,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,165,255),2)

    elif status_mask == "MASK" and status_glass == "GLASSES":
        current_state = "MASK_GLASSES"
        decision_text = "Please remove mask and glasses to enter"

        cv2.putText(frame,"REMOVE MASK & GLASSES",(30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

    elif status_mask == "MASK":
        current_state = "MASK"
        decision_text = "Please remove mask to enter"

        cv2.putText(frame,"MASK DETECTED - ACCESS DENIED",(30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

    elif status_glass == "GLASSES":
        current_state = "GLASSES"
        decision_text = "Please remove glasses to enter"

        cv2.putText(frame,"REMOVE GLASSES",(30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

    elif status_mask == "NO_MASK":
        current_state = "ACCESS"
        decision_text = "Access granted, please welcome"

        cv2.putText(frame,"ACCESS GRANTED",(30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    #  PERFECT VOICE
    if decision_text and current_state:

        now = time.time()

        if current_state != last_state:
            speak(decision_text)
            last_state = current_state
            last_voice_time = now

        elif now - last_voice_time > VOICE_DELAY:
            speak(decision_text)
            last_voice_time = now

    #  DISTANCE TEXT 
    cv2.putText(frame,f"DISTANCE: {distance_status}",
                (30,80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

    #  ATM STATUS PANEL 
    cv2.rectangle(frame,(10,430),(420,470),(30,30,30),-1)

    if decision_text:
        cv2.putText(frame, decision_text[:35],
                    (20,455),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(0,255,255),1)

    #  SCAN LINE 
    h,w,_ = frame.shape
    cv2.line(frame,(0,scan_y),(w,scan_y),(0,0,255),2)

    scan_y += 6
    if scan_y > h:
        scan_y = 0

    cv2.imshow("ATM AI SYSTEM - FINAL PRO",frame)

    if cv2.waitKey(1) & 0xff==ord('q'):
        break
    # if cv2.waitKey(1)==27:
    #     break

cap.release()
cv2.destroyAllWindows()
