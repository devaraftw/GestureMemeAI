import pygame
import sys
import os
import time
import cv2
import numpy as np
import mediapipe as mp

from PIL import Image, ImageSequence

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
)

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


hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.20,
    min_hand_presence_confidence=0.20,
    min_tracking_confidence=0.20,
)


pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker_full.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.30,
    min_pose_presence_confidence=0.30,
    min_tracking_confidence=0.30,
)


# ============================================================
# MEME FILES
# ============================================================

MEMES = {
    "PEACE": "memes/chill_guy.webp",
    "HANDS_HEAD": "memes/hands_head.jpg",
    "CLASPED_HANDS": "memes/clasped_hands.png",
    "POINTING": "memes/open_palm.jpg",
    "ABSOLUTE_CINEMA": "memes/absolute_cinema.jpg",
    "THUMBS_UP": "memes/thumbs_up.jpg",
    "SHH": "memes/shh.gif",
}
SOUNDS = {
    "PEACE": "sounds/peace.mp3",
    "HANDS_HEAD": "sounds/hands_head.mp3",
    "CLASPED_HANDS": "sounds/clasped_hands.mp3",
    "POINTING": "sounds/pointing.mp3",
    "ABSOLUTE_CINEMA": "sounds/absolute_cinema.mp3",
    "THUMBS_UP": "sounds/thumbs_up.mp3",
    "SHH": "sounds/shh.mp3",
}


MEME_NAMES = {
    "PEACE": "Chill Guy",
    "HANDS_HEAD": "Hands Behind Ears",
    "CLASPED_HANDS": "Clasped Hands",
    "POINTING": "Pointing",
    "ABSOLUTE_CINEMA": "Absolute Cinema",
    "THUMBS_UP": "Cat Thumbs Up",
    "SHH": "SHH",
}


# ============================================================
# MEME LOADER
# ============================================================

def load_meme(path):
    """
    Loads both normal images and animated GIFs.

    Returns:
        {
            "frames": [numpy images],
            "durations": [seconds],
            "animated": True/False
        }
    """

    if not os.path.exists(path):
        print("Missing meme:", path)
        return None

    extension = os.path.splitext(path)[1].lower()

    # --------------------------------------------------------
    # GIF
    # --------------------------------------------------------

    if extension == ".gif":

        try:
            gif = Image.open(path)

            frames = []
            durations = []

            for frame in ImageSequence.Iterator(gif):

                rgba = frame.convert("RGBA")

                array = np.array(
                    rgba,
                    dtype=np.uint8
                )

                # RGBA -> BGRA
                bgra = cv2.cvtColor(
                    array,
                    cv2.COLOR_RGBA2BGRA
                )

                frames.append(bgra)

                duration_ms = frame.info.get(
                    "duration",
                    100
                )

                if duration_ms <= 0:
                    duration_ms = 100

                durations.append(
                    duration_ms / 1000.0
                )

            if not frames:
                return None

            return {
                "frames": frames,
                "durations": durations,
                "animated": True,
            }

        except Exception as error:
            print(
                "Could not load GIF:",
                error
            )
            return None

    # --------------------------------------------------------
    # Normal image
    # --------------------------------------------------------

    image = cv2.imread(
        path,
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        print("Could not load:", path)
        return None

    return {
        "frames": [image],
        "durations": [9999.0],
        "animated": False,
    }
def play_sound(gesture):
    """
    Play the sound associated with a gesture.
    """

    path = SOUNDS.get(gesture)

    if path is None:
        return

    if not os.path.exists(path):
        print("Sound file not found:", path)
        return

    try:
        pygame.mixer.music.stop()

        pygame.mixer.music.load(path)

        pygame.mixer.music.play()

    except pygame.error as error:
        print("Could not play sound:", error)


# ============================================================
# GET CURRENT GIF FRAME
# ============================================================

def get_meme_frame(meme, elapsed):
    """
    Returns the correct animation frame based on elapsed time.
    """

    frames = meme["frames"]

    if not meme["animated"]:
        return frames[0]

    durations = meme["durations"]

    total_duration = sum(durations)

    if total_duration <= 0:
        return frames[0]

    current_time = elapsed % total_duration

    accumulated = 0.0

    for index, duration in enumerate(durations):

        accumulated += duration

        if current_time < accumulated:
            return frames[index]

    return frames[-1]


# ============================================================
# MEME OVERLAY
# ============================================================

def overlay_meme(
    frame,
    meme,
    progress,
    elapsed
):
    """
    Places the meme over the webcam.

    Meme opacity:
        88%

    The webcam remains slightly visible underneath.
    """

    if meme is None:
        return frame

    meme_frame = get_meme_frame(
        meme,
        elapsed
    )

    frame_height, frame_width = frame.shape[:2]

    meme_height, meme_width = meme_frame.shape[:2]

    # --------------------------------------------------------
    # Maximum meme size
    # --------------------------------------------------------

    max_width = int(
        frame_width * 0.72
    )

    max_height = int(
        frame_height * 0.72
    )

    scale = min(
        max_width / meme_width,
        max_height / meme_height
    )

    # Pop-in
    scale *= (
        0.78 +
        0.22 * progress
    )

    new_width = max(
        1,
        int(meme_width * scale)
    )

    new_height = max(
        1,
        int(meme_height * scale)
    )

    resized = cv2.resize(
        meme_frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Center
    # --------------------------------------------------------

    x = (
        frame_width -
        new_width
    ) // 2

    y = (
        frame_height -
        new_height
    ) // 2

    x1 = max(
        0,
        x
    )

    y1 = max(
        0,
        y
    )

    x2 = min(
        frame_width,
        x + new_width
    )

    y2 = min(
        frame_height,
        y + new_height
    )

    resized = resized[
        0:y2 - y1,
        0:x2 - x1
    ]

    roi = frame[
        y1:y2,
        x1:x2
    ]

    # --------------------------------------------------------
    # OPACITY
    # --------------------------------------------------------

    alpha = 0.88 * progress

    # --------------------------------------------------------
    # IMAGE WITH ALPHA
    # --------------------------------------------------------

    if resized.ndim == 3 and resized.shape[2] == 4:

        image = resized[:, :, :3]

        image_alpha = (
            resized[:, :, 3]
            / 255.0
        )

        final_alpha = (
            image_alpha *
            alpha
        )

        final_alpha = (
            final_alpha[:, :, None]
        )

        blended = (
            image * final_alpha
            +
            roi * (1 - final_alpha)
        ).astype(np.uint8)

    else:

        blended = cv2.addWeighted(
            resized,
            alpha,
            roi,
            1 - alpha,
            0
        )

    frame[
        y1:y2,
        x1:x2
    ] = blended

    # --------------------------------------------------------
    # Border
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2 - 1, y2 - 1),
        (255, 255, 255),
        2
    )

    return frame


# ============================================================
# QIMAGE
# ============================================================

def frame_to_qimage(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    height, width, channels = rgb.shape

    bytes_per_line = (
        channels * width
    )

    return QImage(
        rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format_RGB888,
    ).copy()


# ============================================================
# MAIN WINDOW
# ============================================================

class GestureMemeWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Gesture Meme AI"
        )

        self.resize(
            1200,
            850
        )

        self.setMinimumSize(
            900,
            650
        )

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        self.detector = GestureDetector()

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():

            print("Could not open webcam")
            raise SystemExit


        self.timestamp = 0

        self.last_gesture = "NONE"

        self.gesture_frames = 0

        self.confirmed_gesture = "NONE"

        self.required_frames = 6

        self.current_meme = None

        self.meme_start_time = 0

        self.lost_frames = 0

        self.remove_after = 12


        # ----------------------------------------------------
        # MEDIAPIPE
        # ----------------------------------------------------

        self.hand_detector = (
            HandLandmarker.create_from_options(
                hand_options
            )
        )

        self.pose_detector = (
            PoseLandmarker.create_from_options(
                pose_options
            )
        )


        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        self.setup_ui()


        # ----------------------------------------------------
        # CAMERA TIMER
        # ----------------------------------------------------

        self.timer = QTimer(
            self
        )

        self.timer.timeout.connect(
            self.update_frame
        )

        self.timer.start(30)


    # ========================================================
    # UI SETUP
    # ========================================================

    def setup_ui(self):

        central = QWidget()

        self.setCentralWidget(
            central
        )

        main_layout = QVBoxLayout(
            central
        )

        main_layout.setContentsMargins(
            22,
            18,
            22,
            18
        )

        main_layout.setSpacing(
            14
        )


        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        header = QHBoxLayout()

        title = QLabel(
            "🎭  GESTURE MEME AI"
        )

        title.setFont(
            QFont(
                "Segoe UI",
                22,
                QFont.Bold
            )
        )

        title.setStyleSheet(
            "color: white;"
        )

        live = QLabel(
            "● LIVE"
        )

        live.setFont(
            QFont(
                "Segoe UI",
                11,
                QFont.Bold
            )
        )

        live.setStyleSheet(
            "color: #65e572;"
        )

        header.addWidget(
            title
        )

        header.addStretch()

        header.addWidget(
            live
        )

        main_layout.addLayout(
            header
        )


        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        camera_frame = QFrame()

        camera_frame.setStyleSheet(
            """
            QFrame {
                background-color: #050505;
                border-radius: 16px;
                border: 1px solid #303030;
            }
            """
        )

        camera_layout = QVBoxLayout(
            camera_frame
        )

        camera_layout.setContentsMargins(
            8,
            8,
            8,
            8
        )


        self.video_label = QLabel()

        self.video_label.setAlignment(
            Qt.AlignCenter
        )

        self.video_label.setMinimumSize(
            700,
            450
        )

        self.video_label.setStyleSheet(
            """
            QLabel {
                background: #000000;
                border-radius: 12px;
            }
            """
        )

        camera_layout.addWidget(
            self.video_label
        )

        main_layout.addWidget(
            camera_frame,
            1
        )


        # ----------------------------------------------------
        # INFO
        # ----------------------------------------------------

        info_frame = QFrame()

        info_frame.setStyleSheet(
            """
            QFrame {
                background-color: #111111;
                border-radius: 14px;
                border: 1px solid #2b2b2b;
            }
            """
        )

        info_layout = QHBoxLayout(
            info_frame
        )

        info_layout.setContentsMargins(
            20,
            14,
            20,
            14
        )


        # Gesture
        gesture_layout = QVBoxLayout()

        gesture_label = QLabel(
            "GESTURE"
        )

        gesture_label.setStyleSheet(
            "color: #888888;"
        )

        self.gesture_value = QLabel(
            "NONE"
        )

        self.gesture_value.setFont(
            QFont(
                "Segoe UI",
                15,
                QFont.Bold
            )
        )

        gesture_layout.addWidget(
            gesture_label
        )

        gesture_layout.addWidget(
            self.gesture_value
        )


        # Meme
        meme_layout = QVBoxLayout()

        meme_label = QLabel(
            "MEME"
        )

        meme_label.setStyleSheet(
            "color: #888888;"
        )

        self.meme_value = QLabel(
            "Waiting..."
        )

        self.meme_value.setFont(
            QFont(
                "Segoe UI",
                15,
                QFont.Bold
            )
        )

        meme_layout.addWidget(
            meme_label
        )

        meme_layout.addWidget(
            self.meme_value
        )


        # Hands
        hands_layout = QVBoxLayout()

        hands_label = QLabel(
            "HANDS"
        )

        hands_label.setStyleSheet(
            "color: #888888;"
        )

        self.hands_value = QLabel(
            "0 / 2"
        )

        self.hands_value.setFont(
            QFont(
                "Segoe UI",
                15,
                QFont.Bold
            )
        )

        hands_layout.addWidget(
            hands_label
        )

        hands_layout.addWidget(
            self.hands_value
        )


        info_layout.addLayout(
            gesture_layout
        )

        info_layout.addSpacing(
            70
        )

        info_layout.addLayout(
            meme_layout
        )

        info_layout.addSpacing(
            70
        )

        info_layout.addLayout(
            hands_layout
        )

        info_layout.addStretch()

        main_layout.addWidget(
            info_frame
        )


        # ----------------------------------------------------
        # FOOTER
        # ----------------------------------------------------

        footer = QLabel(
            "✌️ Peace   •   👍 Thumbs Up   •   "
            "☝️ SHH   •   👉 Pointing   •   "
            "🤲 Clasped   •   🙌 Absolute Cinema"
        )

        footer.setAlignment(
            Qt.AlignCenter
        )

        footer.setStyleSheet(
            """
            QLabel {
                color: #777777;
                padding: 6px;
            }
            """
        )

        main_layout.addWidget(
            footer
        )


        # ----------------------------------------------------
        # DARK THEME
        # ----------------------------------------------------

        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #080808;
            }

            QLabel {
                color: white;
                font-family: "Segoe UI";
            }
            """
        )


    # ========================================================
    # UPDATE FRAME
    # ========================================================

    def update_frame(self):

        success, frame = self.camera.read()

        if not success:
            return

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # MEDIAPIPE IMAGE
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        self.timestamp += 33


        # ----------------------------------------------------
        # HANDS
        # ----------------------------------------------------

        hand_result = (
            self.hand_detector.detect_for_video(
                mp_image,
                self.timestamp
            )
        )


        # ----------------------------------------------------
        # POSE
        # ----------------------------------------------------

        pose_result = (
            self.pose_detector.detect_for_video(
                mp_image,
                self.timestamp
            )
        )


        # ----------------------------------------------------
        # EXTRACT HANDS
        # ----------------------------------------------------

        if hand_result.hand_landmarks:

            hands = (
                hand_result.hand_landmarks
            )

        else:

            hands = []


        # ----------------------------------------------------
        # EXTRACT POSE
        # ----------------------------------------------------

        if pose_result.pose_landmarks:

            pose = (
                pose_result.pose_landmarks[0]
            )

        else:

            pose = None


        # ----------------------------------------------------
        # DETECT GESTURE
        # ----------------------------------------------------

        gesture = self.detector.detect(
            hands,
            pose
        )


        # ----------------------------------------------------
        # HAND COUNT
        # ----------------------------------------------------

        self.hands_value.setText(
            f"{len(hands)} / 2"
        )


        # ====================================================
        # GESTURE STABILITY
        # ====================================================

        if gesture == self.last_gesture:

            self.gesture_frames += 1

        else:

            self.last_gesture = gesture

            self.gesture_frames = 0


        # ====================================================
        # CONFIRM GESTURE
        # ====================================================

        if (
            self.gesture_frames
            >= self.required_frames
        ):

            if (
                self.confirmed_gesture
                != gesture
            ):

                self.confirmed_gesture = (
                    gesture
                )

                print(
                    "Detected:",
                    gesture
                )

                self.gesture_value.setText(
                    gesture.replace(
                        "_",
                        " "
                    )
                )


                # --------------------------------------------
                # LOAD MEME
                # --------------------------------------------

                if gesture in MEMES:

                    meme = load_meme(
                        MEMES[gesture]
                    )

                    if meme is not None:

                        self.current_meme = meme
                        

                        self.meme_start_time = (
                            time.monotonic()
                        )

                        self.meme_value.setText(
                            MEME_NAMES.get(
                                gesture,
                                gesture
                            )
                        )

                        self.lost_frames = 0
                        play_sound(gesture)


        # ====================================================
        # REMOVE MEME
        # ====================================================

        if gesture in (
            "NO_HAND",
            "UNKNOWN"
        ):

            self.lost_frames += 1

        else:

            self.lost_frames = 0


        if (
            self.lost_frames
            >= self.remove_after
        ):

            self.current_meme = None

            self.confirmed_gesture = "NONE"

            self.gesture_value.setText(
                "NONE"
            )

            self.meme_value.setText(
                "Waiting..."
            )


        # ====================================================
        # MEME OVERLAY
        # ====================================================

        if self.current_meme is not None:

            elapsed = (
                time.monotonic()
                -
                self.meme_start_time
            )

            progress = min(
                elapsed / 0.20,
                1.0
            )

            # Smooth pop animation
            progress = (
                1 -
                (1 - progress) ** 3
            )

            frame = overlay_meme(
                frame,
                self.current_meme,
                progress,
                elapsed
            )


        # ====================================================
        # DISPLAY
        # ====================================================

        image = frame_to_qimage(
            frame
        )

        pixmap = QPixmap.fromImage(
            image
        )

        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )


    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(self, event):

        self.timer.stop()

        self.camera.release()

        self.hand_detector.close()

        self.pose_detector.close()

        
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        event.accept()


# ============================================================
# START
# ============================================================

def main():

    app = QApplication(
        sys.argv
    )

    window = GestureMemeWindow()

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()