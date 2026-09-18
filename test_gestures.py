import cv2
import mediapipe as mp

from gesture_detector import GestureDetector


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions


# ============================================================
# HAND DETECTOR
# ============================================================

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.20,
    min_hand_presence_confidence=0.20,
    min_tracking_confidence=0.20
)


# ============================================================
# POSE DETECTOR
# ============================================================

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker_full.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.30,
    min_pose_presence_confidence=0.30,
    min_tracking_confidence=0.30
)


# ============================================================
# GESTURE DETECTOR
# ============================================================

gesture_detector = GestureDetector()


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("Could not open camera")
    raise SystemExit


timestamp = 0


print()
print("==============================")
print("     GESTURE TEST")
print("==============================")
print()
print("Try:")
print("Peace")
print("Hands beside ears")
print("Clasped hands")
print("Pointing toward camera")
print("Absolute Cinema")
print()
print("Press Q to quit.")
print()


# ============================================================
# MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(
    hand_options
) as hand_detector:

    with PoseLandmarker.create_from_options(
        pose_options
    ) as pose_detector:

        while True:

            # ------------------------------------------------
            # CAMERA
            # ------------------------------------------------

            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(
                frame,
                1
            )

            height, width = frame.shape[:2]


            # ------------------------------------------------
            # RGB
            # ------------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )


            # ------------------------------------------------
            # TIMESTAMP
            # ------------------------------------------------

            timestamp += 33


            # ------------------------------------------------
            # DETECT HANDS
            # ------------------------------------------------

            hand_result = hand_detector.detect_for_video(
                mp_image,
                timestamp
            )


            # ------------------------------------------------
            # DETECT BODY
            # ------------------------------------------------

            pose_result = pose_detector.detect_for_video(
                mp_image,
                timestamp
            )


            # ------------------------------------------------
            # GET HANDS
            # ------------------------------------------------

            if hand_result.hand_landmarks:

                hands = hand_result.hand_landmarks

            else:

                hands = []


            # ------------------------------------------------
            # GET POSE
            # ------------------------------------------------

            if pose_result.pose_landmarks:

                pose = pose_result.pose_landmarks[0]

            else:

                pose = None


            # ------------------------------------------------
            # CLASSIFY
            # ------------------------------------------------

            gesture = gesture_detector.detect(
                hands,
                pose
            )


            # =================================================
            # DRAW HANDS
            # =================================================

            connections = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),

                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),

                (0, 9),
                (9, 10),
                (10, 11),
                (11, 12),

                (0, 13),
                (13, 14),
                (14, 15),
                (15, 16),

                (0, 17),
                (17, 18),
                (18, 19),
                (19, 20),

                (5, 9),
                (9, 13),
                (13, 17)
            ]


            for hand in hands:

                for p in hand:

                    x = int(
                        p.x * width
                    )

                    y = int(
                        p.y * height
                    )

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )

                for a, b in connections:

                    x1 = int(
                        hand[a].x * width
                    )

                    y1 = int(
                        hand[a].y * height
                    )

                    x2 = int(
                        hand[b].x * width
                    )

                    y2 = int(
                        hand[b].y * height
                    )

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )


            # =================================================
            # DRAW BODY DEBUG POINTS
            # =================================================

            if pose is not None:

                important = [
                    7,
                    8,
                    11,
                    12,
                    15,
                    16
                ]

                for index in important:

                    p = pose[index]

                    x = int(
                        p.x * width
                    )

                    y = int(
                        p.y * height
                    )

                    cv2.circle(
                        frame,
                        (x, y),
                        7,
                        (255, 0, 0),
                        -1
                    )


            # =================================================
            # DISPLAY
            # =================================================

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Hands: {len(hands)}/2",
                (25, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


            cv2.imshow(
                "Gesture Detector Test",
                frame
            )


            # ------------------------------------------------
            # QUIT
            # ------------------------------------------------

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


camera.release()
cv2.destroyAllWindows()