import cv2, math
import mediapipe as mp
import numpy as np 
from collections import deque


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

def get_finger_states(hand_landmarks):
    """
    Returns which fingers are extended, using joint angles.
    More robust than raw y comparisons.

    Output:
      {
        "thumb":  True/False,
        "index":  True/False,
        "middle": True/False,
        "ring":   True/False,
        "pinky":  True/False,
      }
    """
    lm = hand_landmarks.landmark
    EXTENDED_ANGLE_DEG = 160.0
    EXTENDED_ANGLE_RAD = math.radians(EXTENDED_ANGLE_DEG)

    def is_extended(mcp_idx, pip_idx, tip_idx):
        return _angle_at_joint(lm[mcp_idx], lm[pip_idx], lm[tip_idx]) > EXTENDED_ANGLE_RAD

    # Index: 5 (MCP), 6 (PIP), 8 (TIP)
    index_ext  = is_extended(5, 6, 8)
    # Middle: 9, 10, 12
    middle_ext = is_extended(9, 10, 12)
    # Ring: 13, 14, 16
    ring_ext   = is_extended(13, 14, 16)
    # Pinky: 17, 18, 20
    pinky_ext  = is_extended(17, 18, 20)
    # Thumb: 2 (MCP), 3 (IP), 4 (TIP)
    thumb_ext  = is_extended(2, 3, 4)

    return {
        "thumb":  thumb_ext,
        "index":  index_ext,
        "middle": middle_ext,
        "ring":   ring_ext,
        "pinky":  pinky_ext,
    }

def get_finger_number(hand_landmarks):
    ''' Counts how many fingers are extended.'''

    states = get_finger_states(hand_landmarks)
    thumb  = states["thumb"]
    index  = states["index"]
    middle = states["middle"]
    ring   = states["ring"]
    pinky  = states["pinky"]

    # 1: index only
    if index and not (thumb or middle or ring or pinky):
        return 1

    # 2: index + middle only
    if index and middle and not (thumb or ring or pinky):
        return 2

    # 3: index + middle + ring only
    if index and middle and ring and not (thumb or pinky):
        return 3

    # 4: index + middle + ring + pinky (thumb down)
    if index and middle and ring and pinky and not thumb:
        return 4

    # 5: all five extended
    if index and middle and ring and pinky and thumb:
        return 5

    # Anything else = not a clean number gesture
    return 0

def is_ok_sign(hand_landmarks, dist_ratio_thresh=0.3):
    """
    Detect a rough 'OK' sign:

    - Thumb tip and index tip are very close (forming a circle),
      relative to the hand size.
    - Middle finger is extended (so it's not just a fist).

    This is heuristic, you can tweak thresholds later.
    """
    lm = hand_landmarks.landmark

    thumb_tip = lm[4]
    index_tip = lm[8]

    # Use distance between index MCP and pinky MCP as a rough hand-size reference
    index_mcp = lm[5]
    pinky_mcp = lm[17]

    dx_hand = index_mcp.x - pinky_mcp.x
    dy_hand = index_mcp.y - pinky_mcp.y
    hand_size = math.sqrt(dx_hand * dx_hand + dy_hand * dy_hand)

    if hand_size < 1e-6:
        return False

    # Distance between thumb tip and index tip
    dx = thumb_tip.x - index_tip.x
    dy = thumb_tip.y - index_tip.y
    tip_dist = math.sqrt(dx * dx + dy * dy)

    # Require the two tips to be close relative to hand size
    if tip_dist > dist_ratio_thresh * hand_size:
        return False

    # Also require middle finger extended (just to avoid any random pinch)
    states = get_finger_states(hand_landmarks)
    middle_ext = states["middle"]

    return middle_ext

def is_thumb_up(hand_landmarks):
    """
    Heuristic thumbs-up detector:

    - Thumb tip is ABOVE the base of the index finger (thumb pointing upward)
    - All other fingers are curled (their tips below their PIP joints)
    """

    lm = hand_landmarks.landmark
    states = get_finger_states(hand_landmarks)

    # Thumb must be extended, others mostly folded
    thumb  = states["thumb"]
    index  = states["index"]
    middle = states["middle"]
    ring   = states["ring"]
    pinky  = states["pinky"]

    others_folded = not (index or middle or ring or pinky)

    thumb_tip  = lm[4]
    index_mcp  = lm[5]
    thumb_up_dir = thumb_tip.y < index_mcp.y

    return thumb and others_folded and thumb_up_dir

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

def draw_point_history(image, point_history):
    """
    Draws a fading trail for the pointer (index fingertip).
    point_history: deque of (x, y) or None.
    """
    for idx, point in enumerate(point_history):
        if point is None:
            continue
        x, y = point
        # Older points → smaller / lighter trail; tweak radius if you want
        radius = 2 + idx // 2
        cv2.circle(image, (x, y), radius, (0, 255, 255), 2)
    return image

def detect_circle_direction(point_history,
                            min_points=20,
                            min_radius=25,
                            min_total_angle=2*math.pi*0.8):
    """
    Detect if recent pointer motion forms a roughly circular path and
    estimate its direction:
        - returns 'cw'  for clockwise
        - returns 'ccw' for counterclockwise
        - returns None  if no clear circle

    Uses the pointer trajectory in pixel coords stored in point_history.
    """

    # Keep only actual points (ignore None separators)
    pts = [p for p in point_history if p is not None]
    if len(pts) < min_points:
        return None

    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)

    # Center of the path
    cx = xs.mean()
    cy = ys.mean()

    dx = xs - cx
    dy = ys - cy
    radii = np.sqrt(dx * dx + dy * dy)
    mean_r = radii.mean()

    # Require the circle to be "big enough"
    if mean_r < min_radius:
        return None

    # Radius should be reasonably consistent
    radius_rel_std = radii.std() / (mean_r + 1e-6)
    if radius_rel_std > 0.4:
        return None

    # Angles around the center
    angles = np.arctan2(dy, dx)
    # Unwrap to get smooth evolution over time
    angles_unwrapped = np.unwrap(angles)

    total_angle = angles_unwrapped[-1] - angles_unwrapped[0]

    # Need to cover most of a full rotation
    if abs(total_angle) < min_total_angle:
        return None

    # Screen coords: y increases downward, so this may feel flipped.
    # We define:
    #   total_angle < 0  => 'cw'
    #   total_angle > 0  => 'ccw'
    direction = 'cw' if total_angle < 0 else 'ccw'

    # Clear history so each circle is a single "pulse"
    point_history.clear()

    return direction

    
def detect_opera(mouth_features,
                 open_thresh=0.03,
                 smile_curv_limit=-0.01):
    """
    Decide if 'opera' mouth is active and how open it is.

    Rules:
    - Must be clearly open: mouth_height > open_thresh
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

    return enabled

def detect_swipe_gesture(history, direction="left",
                         min_points=4, min_dx=40, max_dy=80):
    """
    Detect a horizontal swipe in a given direction from recent (x, y) points.

    direction: "left" or "right"
    Returns (is_swipe, dx, dy_span) where:
      dx = end_x - start_x (positive if moved right)
      dy_span = total vertical span
    """

    pts = [p for p in history if p is not None]
    if len(pts) < min_points:
        return False, 0.0, 0.0

    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)

    start_x = xs[0]
    end_x   = xs[-1]

    dx = float(end_x - start_x)       # >0 means moved right
    dy_span = float(ys.max() - ys.min())

    if direction == "left":
        cond = (-dx > min_dx) and (dy_span < max_dy)   # x decreasing a lot
    else:  # "right"
        cond = (dx > min_dx) and (dy_span < max_dy)    # x increasing a lot

    if cond:
        history.clear()  # pulse
        return True, dx, dy_span

    return False, dx, dy_span


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

        self.current_track = None
        self.current_instrument = None

        # Pending selections (wait for confirmation)
        self.pending_track = None
        self.pending_instrument = None

        # OK gesture confirmation
        self.ok_prev = False        # was OK sign present last frame?
        self.ok_pulse = False       # True only on the first frame OK appears

        self.point_history = deque(maxlen=35)  # for smoothing
        self.current_pitch = 0.5
        self.current_volume = 0.5

    
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

            # ===== per-frame state =====
            left_ok_now = False
            right_ok_now = False

            # Flip the frame horizontally so it feels more natural (like a mirror)
            frame = cv2.flip(frame, 1)

            # Convert BGR (OpenCV) -> RGB (MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # -------------- PROCESS HANDS --------------
            results = self.hands.process(frame_rgb)
            thumbs_up_count = 0

            # candidates seen in THIS frame
            right_candidate_track = None
            left_candidate_instrument = None
            pointer_added_this_frame = False

            if results.multi_hand_landmarks:
                h, w, _ = frame.shape

                right_candidate_track = 0
                left_candidate_instrument = 0

                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

                    # Get handedness label ("Left" or "Right")
                    handedness = results.multi_handedness[i]
                    hand_label = handedness.classification[0].label  # "Left" or "Right"

                    # Draw label near wrist
                    wrist = hand_landmarks.landmark[0]
                    cx, cy = int(wrist.x * w), int(wrist.y * h)
                    cv2.putText(
                        frame,
                        hand_label,
                        (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                    # Count fingers for this hand (for track/instrument selection)
                    count = get_finger_number(hand_landmarks)

                    # RIGHT HAND → pending track (1–5, with fist treated as 1)
                    if hand_label == "Right" and count > 1:
                        right_candidate_track = count
                    if hand_label == "Right" and count == 0:
                        right_candidate_track = 1  # fist = 1

                    # LEFT HAND → pending instrument (1–5, with fist treated as 1)
                    if hand_label == "Left" and (count > 1 or count == 0):
                        # fist = 1, else 2–5
                        left_candidate_instrument = (1 if count == 0 else count)

                    # POINTER: any hand with exactly 1 finger (index only)
                    if count == 1:
                        index_tip = hand_landmarks.landmark[8]
                        nx = float(index_tip.x)
                        ny = float(index_tip.y)

                        nx = max(0.0, min(1.0, nx))
                        ny = max(0.0, min(1.0, ny))

                        px_ptr = int(nx * w)
                        py_ptr = int(ny * h)

                        self.point_history.append((px_ptr, py_ptr))
                        pointer_added_this_frame = True

                        # map vertical position to pitch (top=1, bottom=0)
                        self.current_pitch = 1.0 - ny

                        # send event to frontend
                        self.on_event({
                            "type": "pointer_pitch",
                            "x": nx,
                            "y": ny,
                            "pitch": self.current_pitch,
                        })

                    # ✅ OK sign: check independently of count
                    ok_now_this_hand = is_ok_sign(hand_landmarks)
                    if ok_now_this_hand:
                        if hand_label == "Left":
                            left_ok_now = True
                        else:
                            right_ok_now = True

                        cv2.putText(
                            frame,
                            "OK SIGN",
                            (cx - 20, cy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                    # Thumbs up (kept just as a fun debug)
                    if is_thumb_up(hand_landmarks):
                        thumbs_up_count += 1

                # If no pointer this frame, break the trail
                if not pointer_added_this_frame:
                    self.point_history.append(None)

                # Update pending debug text
                if right_candidate_track is not None:
                    self.pending_track = right_candidate_track
                    cv2.putText(
                        frame,
                        f"Pending Track: {self.pending_track}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                if left_candidate_instrument is not None:
                    self.pending_instrument = left_candidate_instrument
                    cv2.putText(
                        frame,
                        f"Pending Instr: {self.pending_instrument}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 200),
                        2
                    )

            else:
                # No hands -> break pointer trail
                self.point_history.append(None)

            # -------------- OK CONTINUOUS DEBUG --------------
            if left_ok_now:
                cv2.putText(
                    frame,
                    "LEFT OK ACTIVE",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )
            if right_ok_now:
                cv2.putText(
                    frame,
                    "RIGHT OK ACTIVE",
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # -------------- DOUBLE THUMBS (optional debug) --------------
            is_double_thumb = (thumbs_up_count >= 2)
            if not hasattr(self, "prev_double_thumb"):
                self.prev_double_thumb = False
            self.double_thumb_pulse = is_double_thumb and not self.prev_double_thumb
            self.prev_double_thumb = is_double_thumb

            if is_double_thumb:
                cv2.putText(
                    frame,
                    "DOUBLE THUMBS UP!",
                    (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            # -------------- PROCESS FACE (key + opera) --------------
            face_results = self.face_mesh.process(frame_rgb)

            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                mouth = get_mouth_features(face_landmarks)

                # Update neutral baseline in the early "neutral" phase
                self.update_neutral_curvature(mouth)

                if self.neutral_done:
                    curv = mouth["curvature"]
                    curv_delta = curv - self.neutral_curvature

                    SMILE_DELTA = -0.01  # tweak as needed

                    if curv_delta < SMILE_DELTA:
                        self.candidate_key = "major"
                    else:
                        self.candidate_key = "minor"
                else:
                    self.candidate_key = "minor"

                # Confirm key whenever ANY OK is being held (either hand)
                if self.neutral_done and (left_ok_now or right_ok_now):
                    if self.candidate_key != self.key_mode:
                        self.key_mode = self.candidate_key
                        self.on_event({
                            "type": "key_mode",
                            "value": self.key_mode
                        })

                cv2.putText(
                    frame,
                    f"Key: {self.key_mode} (cand: {self.candidate_key})",
                    (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

                opera_enabled = detect_opera(mouth)
                self.on_event({
                    "type": "opera_mode",
                    "enabled": opera_enabled,
                })
                cv2.putText(
                    frame,
                    f"opera: {opera_enabled}",
                    (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

            # -------------- APPLY OK-CONFIRMED CHANGES (CONTINUOUS) --------------
            # LEFT OK → confirm TRACK
            if left_ok_now:
                if self.pending_track is not None and self.pending_track != self.current_track:
                    self.current_track = self.pending_track
                    self.on_event({
                        "type": "track_select",
                        "track": self.current_track
                    })

            # RIGHT OK → confirm INSTRUMENT
            if right_ok_now:
                if self.pending_instrument is not None and self.pending_instrument != self.current_instrument:
                    self.current_instrument = self.pending_instrument
                    self.on_event({
                        "type": "instrument_select",
                        "instrument": self.current_instrument
                    })

            # -------------- DRAW CURRENT TRACK / INSTRUMENT DEBUG --------------
            if self.current_track is not None:
                cv2.putText(
                    frame,
                    f"Current Track: {self.current_track}",
                    (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            if self.current_instrument is not None:
                cv2.putText(
                    frame,
                    f"Current Instr: {self.current_instrument}",
                    (10, 280),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 200),
                    2
                )

            # -------------- DRAW POINTER TRAIL + PITCH --------------
            frame = draw_point_history(frame, self.point_history)

            cv2.putText(
                frame,
                f"Pitch: {self.current_pitch:.2f}",
                (10, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 150, 255),
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
