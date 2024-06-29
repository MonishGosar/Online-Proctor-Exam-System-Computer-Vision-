
import cv2 as cv
import mediapipe as mp
import time
import utils, math
import winsound  
import numpy as np


previous_frame = None
motion_threshold = 10  
no_motion_frames = 0

last_message_time = 0
message_display_duration = 10
message_gap_duration = 10

frame_counter =0
CEF_COUNTER =0
TOTAL_BLINKS =0
CLOSED_EYES_FRAME =3
FONTS =cv.FONT_HERSHEY_COMPLEX

eye_position_memory = []  
max_memory_length = 30  
consistent_look_threshold = 10

FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

map_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
camera = cv.VideoCapture(0)


def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    return mesh_coord

def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def blinkRatio(img, landmarks, right_indices, left_indices):
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
  

     
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio 

def eyesExtractor(img, right_eye_coords, left_eye_coords):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    dim = gray.shape

    mask = np.zeros(dim, dtype=np.uint8)

    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    eyes[mask==0]=155
    
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    return cropped_right, cropped_left

def positionEstimator(cropped_eye):
    h, w =cropped_eye.shape
    
    gaussain_blur = cv.GaussianBlur(cropped_eye, (9,9),0)
    median_blur = cv.medianBlur(gaussain_blur, 3)

    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    piece = int(w/3) 

    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece+piece]
    left_piece = threshed_eye[0:h, piece +piece:w]
    
    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

    return eye_position, color 

def pixelCounter(first_piece, second_piece, third_piece):
    right_part = np.sum(first_piece==0)
    center_part = np.sum(second_piece==0)
    left_part = np.sum(third_piece==0)
    eye_parts = [right_part, center_part, left_part]

    max_index = eye_parts.index(max(eye_parts))
    pos_eye ='' 
    if max_index==0:
        pos_eye="RIGHT"
        color=[utils.BLACK, utils.GREEN]
    elif max_index==1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index ==2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye="Closed"
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color


initial_mouth_distance = None  
mouth_distance_set = False  

def mouth_opening_distance(landmarks, upper_lip_indices, lower_lip_indices):
    upper_lip_points = [landmarks[index] for index in upper_lip_indices]
    lower_lip_points = [landmarks[index] for index in lower_lip_indices]
    
    distances = [euclaideanDistance(upper, lower) for upper, lower in zip(upper_lip_points, lower_lip_points)]
    return np.mean(distances)

face_missing_alerted = False


with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

    start_time = time.time()

    while True:
        frame_counter +=1 
        ret, frame = camera.read() 
        if not ret: 
            print("Failed to grab frame")
            cv.waitKey(1000)  
            continue
        
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        faces_detected = face_detection.process(rgb_frame)
        num_faces = len(faces_detected.detections) if faces_detected.detections else 0

        results  = face_mesh.process(rgb_frame)
     
        if results.multi_face_landmarks:
                mesh_coords = landmarksDetection(frame, results, False)

    
                if num_faces > 1:
                    warning_text = "PLS DON'T CHEAT"
                    text_size = cv.getTextSize(warning_text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = (frame_width - text_size[0]) // 2
                    text_y = (frame_height + text_size[1]) // 2
                    cv.putText(frame, warning_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)

                mesh_coords = landmarksDetection(frame, results, False)
                ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

                if ratio >5.5:
                    CEF_COUNTER +=1
                    utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )

                else:
                    if CEF_COUNTER>CLOSED_EYES_FRAME:
                        TOTAL_BLINKS +=1
                        CEF_COUNTER =0
                utils.colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
                
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

                right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                left_coords = [mesh_coords[p] for p in LEFT_EYE]
                crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
            
                eye_position, color = positionEstimator(crop_right)
                utils.colorBackgroundText(frame, f'R: {eye_position}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)
                eye_position_left, color = positionEstimator(crop_left)
                utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8, 8)

                dominant_eye_position = eye_position if eye_position in ["LEFT", "RIGHT"] else eye_position_left
                eye_position_memory.append(dominant_eye_position)

                if len(eye_position_memory) > max_memory_length:
                    eye_position_memory.pop(0)

                current_time = time.time()
                if (eye_position_memory.count("LEFT") > consistent_look_threshold or eye_position_memory.count("RIGHT") > consistent_look_threshold) and (current_time - last_message_time > message_gap_duration):
                    text = "Please look at the screen!"
                    text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    start_x = (frame_width - text_size[0]) // 2
                    start_y = (frame_width - text_size[0]) // 2
                    cv.putText(frame, text, (start_x, start_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    last_message_time = current_time
                    eye_position_memory.clear()
                    winsound.Beep(1000, 1000)  


                if current_time - last_message_time < message_display_duration:
                    text = "Please look at the screen!"
                    text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    start_x = (frame_width - text_size[0]) // 2
                    start_y = (frame_width - text_size[0]) // 2
                    cv.putText(frame, text, (start_x, start_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                border_color = (255, 255, 255)  
                border_thickness = 10
                cv.rectangle(frame, (0, 0), (frame_width, frame_height), border_color, border_thickness)

                text = "NMIMS PROCTOR"
                (text_width, text_height), _ = cv.getTextSize(text, FONTS, 1, 2)
                text_x = (frame_width - text_width) // 2
                text_y = text_height + 20  
                text_color = (0, 0, 0) 
                background_color = (255, 255, 255)  
                cv.rectangle(frame, (text_x, text_y - text_height - 10), (text_x + text_width, text_y + 10), background_color, -1)
                cv.putText(frame, text, (text_x, text_y), FONTS, 1, text_color, 2)


                for index in LIPS:  
                    cv.circle(frame, mesh_coords[index], 1, (0, 255, 0), -1) 
                
                if not mouth_distance_set:
                    initial_mouth_distance = mouth_opening_distance(mesh_coords, UPPER_LIPS, LOWER_LIPS)
                    mouth_distance_set = True

                current_mouth_distance = mouth_opening_distance(mesh_coords, UPPER_LIPS, LOWER_LIPS)
                if current_mouth_distance > initial_mouth_distance * 1.1:  
                    height, width = frame.shape[:2]

                    text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

                    text_x = (width - text_size[0]) // 2
                    text_y = text_size[1] + 100

                    cv.putText(frame, "Mouth Open Detected", (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    

        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key==ord('q') or key ==ord('Q'):
            break
    cv.destroyAllWindows()
    camera.release()
    
