from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture(0)
cap.set(3, 640)  
cap.set(4, 480) 

model = YOLO("yolo-Weights/yolov3-tiny.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                           "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                           "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                           "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                           "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                           "teddy bear", "hair drier", "toothbrush"]


cell_phone_detected = False

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    cell_phone_detected = False

    border_color = (255, 255, 255)  
    border_thickness = 10
    cv2.rectangle(img, (0, 0), (640, 480), border_color, border_thickness)

    proctor_text = "NMIMS PROCTOR"
    text_size = cv2.getTextSize(proctor_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (img.shape[1] - text_size[0]) / 2  
    cv2.putText(img, proctor_text, (int(text_x), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            if classNames[cls] == "cell phone":
                cell_phone_detected = True

    if cell_phone_detected:
        text = "NO PHONES ALLOWED"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (img.shape[1] - text_size[0]) / 2
        text_y = (img.shape[0] + text_size[1]) / 2
        cv2.putText(img, text, (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
