import cv2
import mediapipe as mp

print("Starting...")

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions


print("Creating hand detector...")

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

print("Creating pose detector...")

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker_full.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

print("Opening detectors...")

with HandLandmarker.create_from_options(hand_options) as hand_detector:

    print("Hand detector OK")

    with PoseLandmarker.create_from_options(pose_options) as pose_detector:

        print("Pose detector OK")

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            print("Camera failed")
            raise SystemExit

        print("Camera OK")
        print("Starting camera loop...")

        timestamp = 0

        while True:

            success, frame = camera.read()

            if not success:
                print("Frame failed")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp += 33

            # Hand detection
            hand_result = hand_detector.detect_for_video(
                mp_image,
                timestamp
            )

            # Pose detection
            pose_result = pose_detector.detect_for_video(
                mp_image,
                timestamp
            )

            # Draw hand count
            hand_count = len(
                hand_result.hand_landmarks
            )

            cv2.putText(
                frame,
                f"Hands: {hand_count}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Draw pose landmarks
            if pose_result.pose_landmarks:

                pose = pose_result.pose_landmarks[0]

                important_points = [
                    7,   # left ear
                    8,   # right ear
                    11,  # left shoulder
                    12,  # right shoulder
                    15,  # left wrist
                    16   # right wrist
                ]

                for i in important_points:

                    p = pose[i]

                    x = int(
                        p.x * frame.shape[1]
                    )

                    y = int(
                        p.y * frame.shape[0]
                    )

                    cv2.circle(
                        frame,
                        (x, y),
                        7,
                        (255, 0, 0),
                        -1
                    )

            cv2.imshow(
                "Pose + Hand Test",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        camera.release()
        cv2.destroyAllWindows()

print("Finished")