import cv2, math
import mediapipe as mp
import numpy as np   # you already have cv2, mediapipe, math, etc.

def _angle_at_joint(a, b, c):
    """
    Returns the angle (in radians) at point b formed by points a-b-c.
    a, b, c are MediaPipe landmark objects with x, y, z.
    """
    # Vector BA = A - B
    bax = a.x - b.x
    bay = a.y - b.y
    baz = a.z - b.z

    # Vector BC = C - B
    bcx = c.x - b.x
    bcy = c.y - b.y
    bcz = c.z - b.z

    # dot product and magnitudes
    dot = bax * bcx + bay * bcy + baz * bcz
    mag_ba = math.sqrt(bax*bax + bay*bay + baz*baz)
    mag_bc = math.sqrt(bcx*bcx + bcy*bcy + bcz*bcz)

    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return 0.0

    cos_theta = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.acos(cos_theta)

def count_extended_fingers(hand_landmarks):
    """
    Counts extended fingers (thumb + 4 fingers) based on joint angles.
    More robust to rotation and motion than simple y-comparisons.
    Returns an integer 0–5.
    """
    lm = hand_landmarks.landmark
    fingers = 0

    # Angle threshold: above this -> considered "straight"
    EXTENDED_ANGLE_DEG = 160.0
    EXTENDED_ANGLE_RAD = math.radians(EXTENDED_ANGLE_DEG)

    # Index: 5 (MCP), 6 (PIP), 8 (TIP) -> angle at 6
    if _angle_at_joint(lm[5], lm[6], lm[8]) > EXTENDED_ANGLE_RAD:
        fingers += 1

    # Middle: 9 (MCP), 10 (PIP), 12 (TIP)
    if _angle_at_joint(lm[9], lm[10], lm[12]) > EXTENDED_ANGLE_RAD:
        fingers += 1

    # Ring: 13 (MCP), 14 (PIP), 16 (TIP)
    if _angle_at_joint(lm[13], lm[14], lm[16]) > EXTENDED_ANGLE_RAD:
        fingers += 1

    # Pinky: 17 (MCP), 18 (PIP), 20 (TIP)
    if _angle_at_joint(lm[17], lm[18], lm[20]) > EXTENDED_ANGLE_RAD:
        fingers += 1

    # Thumb: 2 (MCP), 3 (IP), 4 (TIP)
    # Finger is extended if the IP joint angle is also large
    if _angle_at_joint(lm[2], lm[3], lm[4]) > EXTENDED_ANGLE_RAD:
        fingers += 1

    return fingers

def is_thumb_up(hand_landmarks):
    """
    Heuristic thumbs-up detector:

    - Thumb tip is ABOVE the base of the index finger (thumb pointing upward)
    - All other fingers are curled (their tips below their PIP joints)
    """

    lm = hand_landmarks.landmark

    # Landmarks we care about
    thumb_tip  = lm[4]
    index_mcp  = lm[5]   # base of index finger

    # Thumb up: tip higher (smaller y) than index MCP
    thumb_up = thumb_tip.y < index_mcp.y

    # Other 4 fingers: check they are NOT extended
    index_up  = lm[8].y  < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up   = lm[16].y < lm[14].y
    pinky_up  = lm[20].y < lm[18].y

    others_folded = not (index_up or middle_up or ring_up or pinky_up)

    return thumb_up and others_folded

def euclidean_dist(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def get_mouth_features(face_landmarks):
    lm = face_landmarks.landmark

    left_corner  = lm[61]
    right_corner = lm[291]
    upper_lip    = lm[13]
    lower_lip    = lm[14]

    mouth_width = euclidean_dist(left_corner, right_corner)
    mouth_height = euclidean_dist(upper_lip, lower_lip)

    center_y = (upper_lip.y + lower_lip.y) / 2
    corners_y = (left_corner.y + right_corner.y) / 2
    curvature = corners_y - center_y

    if mouth_width > 1e-6:
        shape_ratio = mouth_height / mouth_width
    else:
        shape_ratio = 0.0

    return {
        "mouth_width": mouth_width,
        "mouth_height": mouth_height,
        "curvature": curvature,
        "shape_ratio": shape_ratio,
    }

def classify_expression(mouth_features, smile_thresh=-0.01, frown_thresh=0.01):
    """
    Very simple: based on curvature only.
    curvature < smile_thresh => 'smile'
    curvature > frown_thresh => 'frown'
    else => 'neutral'
    """
    curv = mouth_features["curvature"]
    if curv < smile_thresh:
        return "smile"
    elif curv > frown_thresh:
        return "frown"
    else:
        return "neutral"
    
def detect_opera(mouth_features,
                 open_thresh=0.03,
                 max_open=0.09,
                 smile_curv_limit=-0.01):
    """
    Decide if 'opera' mouth is active and how open it is.

    Rules:
    - Must be clearly open: mouth_height > open_thresh
    - Must NOT be a big smile: curvature must not be strongly negative
      (curvature < 0 = smile; more negative = bigger smile)
    - Shape doesn't need to be perfectly round.

    Returns:
        enabled (bool), open_ratio (0..1)
    """
    h = mouth_features["mouth_height"]
    curv = mouth_features["curvature"]

    # 1) Open enough vertically
    is_open_enough = h > open_thresh

    # 2) Not a huge smile (filter out smile-with-teeth)
    not_big_smile = curv > smile_curv_limit
    # e.g. -0.05 is big smile, -0.005 is mild → we only reject big ones

    enabled = is_open_enough and not_big_smile

    # 3) Continuous openness for pitch: map h into [0, 1]
    if not enabled:
        open_ratio = 0.0
    else:
        # clamp between open_thresh and max_open
        clamped = max(open_thresh, min(h, max_open))
        open_ratio = (clamped - open_thresh) / (max_open - open_thresh)

    return enabled, open_ratio

class MotionEngine:
    def __init__(self, on_event=None):
        self.on_event = on_event or (lambda e: print("EVENT:", e))

        # Mediapipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.neutral_curvature = 0.0
        self.neutral_samples = []
        self.neutral_frames = 0
        self.neutral_done = False

        # Key mode state: we start in MINOR (no smile yet)
        self.key_mode = "minor"        # committed key: "major" or "minor"
        self.candidate_key = "minor"   # based on current smile
        
        self.prev_double_thumb = False
        self.double_thumb_pulse = False


    def update_neutral_curvature(self, mouth_features,
                                max_open_height=0.015,
                                max_abs_curv=0.02,
                                needed_frames=30):
        """
        Learn the user's neutral mouth curvature from early frames:
        - Only when mouth is not very open
        - And not a big smile or big frown
        """
        if self.neutral_done:
            return

        h = mouth_features["mouth_height"]
        curv = mouth_features["curvature"]

        # Only treat as neutral if mouth is fairly closed & not extreme curvature
        if h < max_open_height and abs(curv) < max_abs_curv:
            self.neutral_samples.append(curv)
            self.neutral_frames += 1

        if self.neutral_frames >= needed_frames:
            self.neutral_curvature = sum(self.neutral_samples) / len(self.neutral_samples)
            self.neutral_done = True
            # Optional: send an event to frontend that calibration is done
            self.on_event({
                "type": "expression_calibration",
                "status": "done"
            })
   
    def run(self):
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: cannot open camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame horizontally so it feels more natural (like a mirror)
            frame = cv2.flip(frame, 1)

            # Convert BGR (OpenCV) -> RGB (MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # -------------- PROCESS HANDS --------------
            results = self.hands.process(frame_rgb)
            thumbs_up_count = 0
            max_fingers = 0

            # If hands are detected, draw landmarks and count fingers
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                           
                    finger_count = count_extended_fingers(hand_landmarks)

                    # Track select: use max finger count
                    if finger_count > max_fingers:
                        max_fingers = finger_count
                    
                    # Thumbs up on this hand?
                    if is_thumb_up(hand_landmarks):
                        thumbs_up_count += 1
                    
                # Emit an event if 1–4 fingers are up (you can extend to 5 when thumb logic is added)
                if max_fingers > 0:
                    self.on_event({
                        "type": "track_select",
                        "track": max_fingers
                    })
                
                # Draw the count on the frame
                cv2.putText(
                    frame,
                    f"Fingers: {finger_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            is_double_thumb = (thumbs_up_count >= 2)

            # Make this a "pulse" (true only on the first frame it appears)
            if not hasattr(self, "prev_double_thumb"):
                self.prev_double_thumb = False

            # Pulse = True only on the first frame where we see 2 thumbs up
            self.double_thumb_pulse = is_double_thumb and not self.prev_double_thumb
            self.prev_double_thumb = is_double_thumb

            # debug text
            if is_double_thumb:
                cv2.putText(
                    frame,
                    "DOUBLE THUMBS UP!",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
       
            # -------------- PROCESS FACE --------------#   
            face_results = self.face_mesh.process(frame_rgb)

            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                mouth = get_mouth_features(face_landmarks)
                
                # Update neutral baseline in the early "neutral" phase
                self.update_neutral_curvature(mouth)

                if self.neutral_done:
                    
                    curv = mouth["curvature"]
                    curv_delta = curv - self.neutral_curvature

                    # Tune this threshold: more negative = stronger smile needed
                    SMILE_DELTA = -0.01

                    if curv_delta < SMILE_DELTA:
                        # Big smile -> MAJOR
                        self.candidate_key = "major"
                    else:
                        # Neutral or corners down -> MINOR
                        self.candidate_key = "minor"
                else:
                    # While calibrating, we can just stay in MINOR
                    self.candidate_key  = "minor"

                # If key changed, lock in new state and send event once
                if self.neutral_done and self.double_thumb_pulse:
                    if self.candidate_key != self.key_mode:
                        self.key_mode = self.candidate_key
                        self.on_event({
                            "type": "key_mode",
                            "value": self.key_mode   # "major" or "minor"
                        })
                            
                cv2.putText(
                    frame,
                    f"Key: {self.key_mode} (cand: {self.candidate_key})",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

                # --- OPERA DETECTION ---
                opera_enabled, open_ratio = detect_opera(mouth)
                self.on_event({
                    "type": "opera_mode",
                    "enabled": opera_enabled,
                    "openness": open_ratio   # 0..1, how big the mouth is
                })
                cv2.putText(
                    frame,
                    f"opera: {opera_enabled} open={open_ratio:.2f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

            # --- SHOW FRAME + EXIT KEY ---
            cv2.imshow("Motion Debug", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



def main():
    engine = MotionEngine()
    engine.run()


if __name__ == "__main__":
    main()
