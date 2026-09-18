import math


class GestureDetector:

    # ========================================================
    # BASIC MATH
    # ========================================================

    @staticmethod
    def distance_2d(a, b):
        return math.sqrt(
            (a.x - b.x) ** 2 +
            (a.y - b.y) ** 2
        )

    @staticmethod
    def distance_3d(a, b):
        return math.sqrt(
            (a.x - b.x) ** 2 +
            (a.y - b.y) ** 2 +
            (a.z - b.z) ** 2
        )

    @staticmethod
    def angle_3_points(a, b, c):
        ba = (
            a.x - b.x,
            a.y - b.y,
            a.z - b.z
        )

        bc = (
            c.x - b.x,
            c.y - b.y,
            c.z - b.z
        )

        dot = (
            ba[0] * bc[0]
            + ba[1] * bc[1]
            + ba[2] * bc[2]
        )

        mag_ba = math.sqrt(
            ba[0] ** 2
            + ba[1] ** 2
            + ba[2] ** 2
        )

        mag_bc = math.sqrt(
            bc[0] ** 2
            + bc[1] ** 2
            + bc[2] ** 2
        )

        if mag_ba == 0 or mag_bc == 0:
            return 0.0

        value = dot / (mag_ba * mag_bc)
        value = max(-1.0, min(1.0, value))

        return math.degrees(
            math.acos(value)
        )

    # ========================================================
    # FINGER DETECTION
    # ========================================================

    def finger_is_extended(
        self,
        hand,
        mcp,
        pip,
        dip,
        tip
    ):
        angle1 = self.angle_3_points(
            hand[mcp],
            hand[pip],
            hand[dip]
        )

        angle2 = self.angle_3_points(
            hand[pip],
            hand[dip],
            hand[tip]
        )

        return (
            angle1 > 140
            and angle2 > 140
        )

    def get_fingers(self, hand):
        """
        Returns:
        [index, middle, ring, pinky]
        """

        return [
            self.finger_is_extended(
                hand, 5, 6, 7, 8
            ),
            self.finger_is_extended(
                hand, 9, 10, 11, 12
            ),
            self.finger_is_extended(
                hand, 13, 14, 15, 16
            ),
            self.finger_is_extended(
                hand, 17, 18, 19, 20
            )
        ]

    def is_open_hand(self, hand):
        return all(
            self.get_fingers(hand)
        )

    # ========================================================
    # THUMBS UP
    # ========================================================

    def is_thumbs_up(self, hand):
        """
        Detect 👍

        Only works when exactly one hand is present,
        because detect() only calls this for one hand.
        """

        fingers = self.get_fingers(hand)

        # Other four fingers must be folded.
        if any(fingers):
            return False

        thumb_tip = hand[4]
        thumb_ip = hand[3]
        thumb_mcp = hand[2]
        wrist = hand[0]

        thumb_extended = (
            self.distance_3d(
                thumb_tip,
                wrist
            )
            >
            self.distance_3d(
                thumb_ip,
                wrist
            ) * 1.12
        )

        if not thumb_extended:
            return False

        # Thumb points upward.
        return (
            thumb_tip.y < thumb_ip.y - 0.04
            and
            thumb_tip.y < thumb_mcp.y - 0.08
        )

    # ========================================================
    # PEACE
    # ========================================================

    def is_peace(self, hand):
        """
        Detect ✌️
        """

        fingers = self.get_fingers(hand)

        return (
            fingers[0]
            and fingers[1]
            and not fingers[2]
            and not fingers[3]
        )

    # ========================================================
    # SHH / ONE FINGER
    # ========================================================

    def is_shh(self, hand):
        """
        Detect ☝️

        Requirements:
        - index finger extended
        - middle/ring/pinky folded
        - index finger roughly vertical

        This is intentionally different from POINTING.
        """

        fingers = self.get_fingers(hand)

        index = fingers[0]
        middle = fingers[1]
        ring = fingers[2]
        pinky = fingers[3]

        # Only index finger should be extended.
        if not (
            index
            and not middle
            and not ring
            and not pinky
        ):
            return False

        wrist = hand[0]
        index_mcp = hand[5]
        index_tip = hand[8]

        # Direction from index base to fingertip.
        dx = index_tip.x - index_mcp.x
        dy = index_tip.y - index_mcp.y

        # Avoid division by zero.
        length = math.sqrt(
            dx * dx +
            dy * dy
        )

        if length == 0:
            return False

        # How vertical is the finger?
        vertical_ratio = abs(dy) / length

        # A vertical finger should have a high ratio.
        is_vertical = (
            vertical_ratio > 0.75
        )

        # Fingertip should be above the wrist.
        tip_is_high = (
            index_tip.y <
            wrist.y - 0.08
        )

        return (
            is_vertical
            and tip_is_high
        )

    # ========================================================
    # POINTING TOWARD CAMERA
    # ========================================================

    def is_pointing(self, hand):
        """
        Detect the pointing-toward-camera pose.
        """

        fingers = self.get_fingers(hand)

        # All four fingers must be extended.
        if not all(fingers):
            return False

        tips = [
            hand[8],
            hand[12],
            hand[16],
            hand[20]
        ]

        x_spread = (
            max(p.x for p in tips)
            -
            min(p.x for p in tips)
        )

        y_spread = (
            max(p.y for p in tips)
            -
            min(p.y for p in tips)
        )

        # Fingers should appear grouped.
        if x_spread > 0.20:
            return False

        if y_spread > 0.32:
            return False

        palm_z = (
            hand[0].z
            + hand[5].z
            + hand[9].z
            + hand[13].z
            + hand[17].z
        ) / 5

        fingers_z = (
            hand[8].z
            + hand[12].z
            + hand[16].z
            + hand[20].z
        ) / 4

        fingers_closer = (
            palm_z - fingers_z
        ) > 0.03

        return fingers_closer

    # ========================================================
    # CLASPED HANDS
    # ========================================================

    def is_clasped(
        self,
        hand1,
        hand2
    ):
        points = [
            0,
            5,
            9,
            13,
            17,
            8,
            12,
            16,
            20
        ]

        closest_distance = 999.0

        for p1 in points:

            for p2 in points:

                distance = self.distance_2d(
                    hand1[p1],
                    hand2[p2]
                )

                closest_distance = min(
                    closest_distance,
                    distance
                )

        return (
            closest_distance < 0.13
        )

    # ========================================================
    # POSE HELPERS
    # ========================================================

    @staticmethod
    def get_pose_points(pose):
        return {
            "left_ear": pose[7],
            "right_ear": pose[8],
            "left_shoulder": pose[11],
            "right_shoulder": pose[12],
            "left_wrist": pose[15],
            "right_wrist": pose[16]
        }

    # ========================================================
    # HANDS BESIDE EARS
    # ========================================================

    def is_hands_head(
        self,
        hands,
        pose
    ):
        if len(hands) < 2:
            return False

        if pose is None:
            return False

        points = self.get_pose_points(
            pose
        )

        ears = sorted(
            [
                points["left_ear"],
                points["right_ear"]
            ],
            key=lambda p: p.x
        )

        wrists = sorted(
            [
                hands[0][0],
                hands[1][0]
            ],
            key=lambda p: p.x
        )

        left_distance = self.distance_2d(
            wrists[0],
            ears[0]
        )

        right_distance = self.distance_2d(
            wrists[1],
            ears[1]
        )

        return (
            left_distance < 0.30
            and
            right_distance < 0.30
        )

    # ========================================================
    # ABSOLUTE CINEMA
    # ========================================================

    def is_absolute_cinema(
        self,
        hands,
        pose
    ):
        if len(hands) < 2:
            return False

        if pose is None:
            return False

        # Both hands must be open.
        if not self.is_open_hand(hands[0]):
            return False

        if not self.is_open_hand(hands[1]):
            return False

        points = self.get_pose_points(
            pose
        )

        wrists = sorted(
            [
                points["left_wrist"],
                points["right_wrist"]
            ],
            key=lambda p: p.x
        )

        shoulders = sorted(
            [
                points["left_shoulder"],
                points["right_shoulder"]
            ],
            key=lambda p: p.x
        )

        # Wrists above shoulders.
        if not (
            wrists[0].y <
            shoulders[0].y - 0.03
            and
            wrists[1].y <
            shoulders[1].y - 0.03
        ):
            return False

        # Wrists outside shoulders.
        if not (
            wrists[0].x <
            shoulders[0].x - 0.05
            and
            wrists[1].x >
            shoulders[1].x + 0.05
        ):
            return False

        shoulder_width = (
            shoulders[1].x -
            shoulders[0].x
        )

        wrist_width = (
            wrists[1].x -
            wrists[0].x
        )

        if shoulder_width <= 0:
            return False

        return (
            wrist_width /
            shoulder_width
        ) >= 1.30

    # ========================================================
    # MAIN DETECTOR
    # ========================================================

    def detect(
        self,
        hands,
        pose
    ):
        """
        Gesture rules:

        ONE HAND:
            THUMBS_UP
            PEACE
            SHH
            POINTING

        TWO HANDS:
            ABSOLUTE_CINEMA
            CLASPED_HANDS
            HANDS_HEAD
        """

        # ----------------------------------------------------
        # NO HAND
        # ----------------------------------------------------

        if len(hands) == 0:
            return "NO_HAND"

        # ----------------------------------------------------
        # TWO HANDS
        # ----------------------------------------------------

        if len(hands) >= 2:

            if self.is_absolute_cinema(
                hands,
                pose
            ):
                return "ABSOLUTE_CINEMA"

            if self.is_clasped(
                hands[0],
                hands[1]
            ):
                return "CLASPED_HANDS"

            if self.is_hands_head(
                hands,
                pose
            ):
                return "HANDS_HEAD"

            # Never use one-hand gestures here.
            return "UNKNOWN"

        # ----------------------------------------------------
        # EXACTLY ONE HAND
        # ----------------------------------------------------

        hand = hands[0]

        # IMPORTANT:
        # SHH must be checked before anything that could
        # potentially classify a similar pose.

        if self.is_shh(hand):
            return "SHH"

        if self.is_thumbs_up(hand):
            return "THUMBS_UP"

        if self.is_peace(hand):
            return "PEACE"

        if self.is_pointing(hand):
            return "POINTING"

        return "UNKNOWN"