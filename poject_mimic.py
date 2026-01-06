import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

face = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

def midpoint(p1, p2):
    return ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_res = pose.process(rgb)
    face_res = face.process(rgb)
    hand_res = hands.process(rgb)

    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    points = {}

    #body
    if pose_res.pose_landmarks:
        for i, lm in enumerate(pose_res.pose_landmarks.landmark):
            cx, cy = int(lm.x*w), int(lm.y*h)
            points[i] = (cx, cy)

        mp_draw.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        ls = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = points[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = points[mp_pose.PoseLandmark.RIGHT_HIP.value]

        neck = midpoint(ls, rs)
        pelvis = midpoint(lh, rh)
        spine = midpoint(neck, pelvis)

        cv2.line(dummy, neck, spine, (255,255,255), 3)
        cv2.line(dummy, spine, pelvis, (255,255,255), 3)

        for a, b in mp_pose.POSE_CONNECTIONS:
            if a in points and b in points:
                cv2.line(dummy, points[a], points[b], (255,255,255), 3)

    #face
    if face_res.multi_face_landmarks:
        fl = face_res.multi_face_landmarks[0].landmark
        nose = (int(fl[1].x*w), int(fl[1].y*h))
        le = (int(fl[33].x*w), int(fl[33].y*h))
        re = (int(fl[263].x*w), int(fl[263].y*h))
        head = midpoint(le, re)

        cv2.circle(dummy, head, 6, (0,0,255), -1)
        cv2.line(dummy, head, nose, (0,0,255), 3)

    #hands
    if hand_res.multi_hand_landmarks:
        for hand in hand_res.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                dummy,
                hand,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255,255,255), thickness=2)
            )

    combined = np.hstack((frame, dummy))
    cv2.imshow("Mister mime", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()