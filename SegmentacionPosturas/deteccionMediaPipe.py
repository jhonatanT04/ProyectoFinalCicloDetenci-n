import cv2
import mediapipe as mp
import time
import time
import numpy as np

# MediaPipe

# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# pose = mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

def deteccionPose(frame,boxes, weights):
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for (x, y, w, h) in boxes:
        # Recortar región de la persona
        person_roi = rgb[y:y+h, x:x+w]
        
        # Crear instancia temporal de Pose para cada persona
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            results = pose.process(person_roi)
            
            if results.pose_landmarks:
                # Ajustar coordenadas al frame completo
                for landmark in results.pose_landmarks.landmark:
                    landmark.x = (landmark.x * w + x) / frame.shape[1]
                    landmark.y = (landmark.y * h + y) / frame.shape[0]
                
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
    
    return frame


def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def np_angle(a, b, c):
    ba = a - b
    bc = c - b

    cross = ba[0]*bc[1] - ba[1]*bc[0]  
    dot = np.dot(ba, bc)

    angle = np.degrees(np.arctan2(cross, dot))
    return angle
def calculate_angle(shoulder_center, hip_center):
    dy = shoulder_center[1] - hip_center[1]
    dx = shoulder_center[0] - hip_center[0]
    angle = np.arctan2(dy, dx)
    return abs(np.degrees(angle))

def classify_pose(p,mp_pose):
    
    left_sh = np.array([p[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        p[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
    right_sh = np.array([p[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                         p[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])

    left_elbow  = np.array([p[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                        p[mp_pose.PoseLandmark.LEFT_ELBOW].y])
    right_elbow = np.array([p[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                            p[mp_pose.PoseLandmark.RIGHT_ELBOW].y])

    left_wrist  = np.array([p[mp_pose.PoseLandmark.LEFT_WRIST].x,
                        p[mp_pose.PoseLandmark.LEFT_WRIST].y])
    right_wrist = np.array([p[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                            p[mp_pose.PoseLandmark.RIGHT_WRIST].y])
    
    
    elbow_dist = euclidean(left_elbow, right_elbow)
    shoulder_dist = euclidean(left_sh, right_sh)

    normalized_elbow_dist = elbow_dist / shoulder_dist
    
    
    left_hip  = np.array([p[mp_pose.PoseLandmark.LEFT_HIP].x,
                        p[mp_pose.PoseLandmark.LEFT_HIP].y])
    right_hip = np.array([p[mp_pose.PoseLandmark.RIGHT_HIP].x,
                            p[mp_pose.PoseLandmark.RIGHT_HIP].y])

    left_knee = np.array([p[mp_pose.PoseLandmark.LEFT_KNEE].x,
                        p[mp_pose.PoseLandmark.LEFT_KNEE].y])
    right_knee = np.array([p[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                         p[mp_pose.PoseLandmark.RIGHT_KNEE].y])
    
    left_ankle= np.array([p[mp_pose.PoseLandmark.LEFT_ANKLE].x, p[mp_pose.PoseLandmark.LEFT_ANKLE].y])
    right_ankle= np.array([p[mp_pose.PoseLandmark.RIGHT_ANKLE].x, p[mp_pose.PoseLandmark.RIGHT_ANKLE].y])

    hip_dist = euclidean(left_hip, right_hip)
    knee_dist_left = euclidean(left_knee, left_hip)
    knee_dist_right = euclidean(right_knee, right_hip)
    
    normalized_hip_dist = ((knee_dist_left+knee_dist_right)/2) / hip_dist
    
    
    
    
    wrist_dist = euclidean(left_wrist, right_wrist)
    
    normalized_wrist_dist = wrist_dist / shoulder_dist
    
    # Angulo con respecto a los brazos, codos y manos
    
    angulo_left = np_angle(left_sh,left_elbow,left_wrist)
    angulo_right = np_angle(right_sh,right_elbow,right_wrist)
    
    # print(angulo_left)
    # (angulo_left and angulo_right ) == (160-180)
    # print(f"angulos= ({angulo_right:.2f} || {angulo_left:.2f})  ")
    
    # Angulo de el centro de los hombors y torso con respecto al eje x 
    sh_center = (left_sh+right_sh)/2
    hip_center = (left_hip+right_hip)/2
    torso_angle = calculate_angle(sh_center,hip_center)
    print("==="*10)
    print(torso_angle)
    # anulo bueno 70 - 110
    if(110 > torso_angle > 70):    
        if (normalized_elbow_dist >= 1.1 and normalized_elbow_dist < 2.3):
        
            return "Pose Normal"
        elif normalized_elbow_dist <= 1.1 :
        
            if normalized_wrist_dist <= 0.25 and ((angulo_right<40 and angulo_right>10)and (angulo_left<-10 and angulo_left>-40)):
                return "Amen"
            elif(normalized_wrist_dist > 0.25 and((angulo_right<100 and angulo_right>70)and (angulo_left<-70 and angulo_left>-100))):
                return "Brazo cruzado"
            
            return "normal"
        
        elif (normalized_elbow_dist >=2.30):
            if (angulo_left>=160 or angulo_left<=-160) and (angulo_right>=160 or angulo_right<=-160):
                return "Forma T"
            elif(angulo_left<=100 and angulo_left>=50) and (angulo_right<=-50 and angulo_right>=-100):
                return "KAKA"
            return "Nomaml"
    # # Torso precaucion (45) 70-50 or 110-130 
    # elif (70 > torso_angle >50) or (110<torso_angle<130):
        
        
    #     return "Precauncion"
        
    # elif (50 > torso_angle >20) or (130<torso_angle<160):
        
    #     return "Flexion "    
    # # Torso suelo (0)
    # elif(20 > torso_angle >0 or 180>torso_angle >160):
    #     return "Se callo"
    
    return "DESCONOCIDO"