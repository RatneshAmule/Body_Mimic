import cv2
import mediapipe as mp
import numpy as np

# iNIT 
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
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

cap = cv2.VideoCapture(0)

# UTILS 
def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

#  LOOP 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_res = pose.process(rgb)
    face_res = face.process(rgb)

    dummy = np.zeros((h, w, 3), dtype=np.uint8)

    points = {}

    # BoDY poisture
    if pose_res.pose_landmarks:
        for i, lm in enumerate(pose_res.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            points[i] = (cx, cy)

        # Skeleton
        mp_draw.draw_landmarks(
            frame,
            pose_res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # joints
        ls = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = points[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = points[mp_pose.PoseLandmark.RIGHT_HIP.value]

        neck = midpoint(ls, rs)
        pelvis = midpoint(lh, rh)
        spine = midpoint(neck, pelvis)

        # virtual joints
        for p in [neck, pelvis, spine]:
            cv2.circle(dummy, p, 6, (0, 255, 255), -1)

        cv2.line(dummy, neck, spine, (255, 255, 255), 3)
        cv2.line(dummy, spine, pelvis, (255, 255, 255), 3)

        # draw body joints
        for a, b in mp_pose.POSE_CONNECTIONS:
            if a in points and b in points:
                cv2.line(dummy, points[a], points[b], (255, 255, 255), 3)

    #  FACE and HEAD 
    if face_res.multi_face_landmarks:
        fl = face_res.multi_face_landmarks[0].landmark

        def fp(i):
            return (int(fl[i].x * w), int(fl[i].y * h))

        nose = fp(1)
        le = fp(33)
        re = fp(263)
        head_center = midpoint(le, re)

        # head direction
        cv2.circle(dummy, head_center, 8, (0, 0, 255), -1)
        cv2.line(dummy, head_center, nose, (0, 0, 255), 3)

    # DISPLAY 
    combined = np.hstack((frame, dummy))
    cv2.imshow("REAL vs FULL BODY MIMIC (ACCURATE)", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
