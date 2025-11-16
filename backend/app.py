from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from motion_track import MotionEngine, get_finger_number, is_ok_sign, is_thumb_up, is_fist, is_rock_sign

app = Flask(__name__)
CORS(app)

# Initialize motion engine ONCE (not per request)
motion_engine = MotionEngine(on_event=lambda e: None)  # Disable event printing for speed

MAX_WIDTH = 320

@app.route('/api/motion/process-frame', methods=['POST'])
def process_frame():
    """
    Receive a base64-encoded frame from frontend,
    process it with motion tracking, return results.
    """
    try:
        data = request.get_json()
        
        if 'frame' not in data:
            return jsonify({'error': 'No frame provided'}), 400
        
        # Fast decode base64 image
        img_data = base64.b64decode(data['frame'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode frame'}), 400
        
        # Process frame with motion engine
        h, w, _ = frame.shape
        if w > MAX_WIDTH:
            scale = MAX_WIDTH / float(w)
            new_size = (MAX_WIDTH, int(h * scale))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)

        results = process_single_frame(frame)
        return jsonify(results)
    
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

def process_single_frame(frame):
    """Process a single frame and return motion tracking results."""
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    h, w, _ = frame.shape

    # Make sure these attrs exist (they *are* in MotionEngine.__init__, but safe)
    if not hasattr(motion_engine, "current_volume"):
        motion_engine.current_volume = 0.7  # Start at 70% volume instead of 0
    if not hasattr(motion_engine, "is_playing"):
        motion_engine.is_playing = True
    # if not hasattr(motion_engine, "prev_two_fists"):
    #     motion_engine.prev_two_fists = False
    if not hasattr(motion_engine, "current_instrument"):
        motion_engine.current_instrument = None
    if not hasattr(motion_engine, "pending_instrument"):
        motion_engine.pending_instrument = None
    if not hasattr(motion_engine, "prev_rock_any"):
        motion_engine.prev_rock_any = False
    
    # Process hands - handle MediaPipe timestamp errors
    try:
        hands_results = motion_engine.hands.process(frame_rgb)
    except Exception as mp_error:
        # MediaPipe timestamp errors are non-fatal, just skip this frame
        if "timestamp mismatch" in str(mp_error).lower():
            print(f"Skipping frame due to timestamp issue")
            # Return previous state without updating
            return {
                'hands': [],
                'gestures': {},
                'current_instrument': getattr(motion_engine, 'current_instrument', None),
                'prev_rock_any': motion_engine.prev_rock_any,
                'pointer': None,
                'volume': motion_engine.current_volume,
                'play_toggle': False,
                'playing': bool(motion_engine.is_playing),
                'shuffle_triggered': False,
            }
        else:
            raise  # Re-raise if it's a different error
    
    #face_results = motion_engine.face_mesh.process(frame_rgb)
    
    response = {
        'hands': [],
        'gestures': {},
        'current_instrument': getattr(motion_engine, 'current_instrument', None),
        'prev_rock_any': motion_engine.prev_rock_any,
        # 'prev_two_fists': motion_engine.prev_two_fists,
        'pointer': None,
        'volume': motion_engine.current_volume,
        'play_toggle': False,
        'playing': bool(motion_engine.is_playing),
        'shuffle_triggered': False,
    }

    # fist_count = 0
    rock_now = False
    # left_wrist_y = None
    # right_wrist_y = None
    
    # Extract hand landmarks and gestures
    if hands_results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            handedness = hands_results.multi_handedness[i]
            hand_label = handedness.classification[0].label
            
            # Convert landmarks to list of dicts
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append({
                    'x': float(lm.x),
                    'y': float(lm.y),
                    'z': float(lm.z)
                })
            
            # Detect gestures
            finger_count = get_finger_number(hand_landmarks)
            is_ok = is_ok_sign(hand_landmarks)
            # is_thumbs_up = is_thumb_up(hand_landmarks)
            # is_fist_gesture = is_fist(hand_landmarks)
            is_rock = is_rock_sign(hand_landmarks)
            
            # Track rock gesture for shuffle
            if is_rock:
                rock_now = True
            
            # Track fists for play/pause
            # if is_fist_gesture:
            #     fist_count += 1

            # Update pending selections based on hand (RIGHT HAND)
            if hand_label == "Right" and 4 >= finger_count > 0:
                motion_engine.pending_instrument = finger_count
            
            # Confirm with OK gesture (LEFT HAND confirms)
            if hand_label == "Left" and is_ok:
                if motion_engine.pending_instrument is not None:
                    motion_engine.current_instrument = motion_engine.pending_instrument
            
            # Volume pointer only for LEFT hand with 1 finger (when not doing OK gesture)
            if hand_label == "Left" and finger_count == 1 and not is_ok:
                index_tip = hand_landmarks.landmark[8]
                ny = float(index_tip.y)

                # Map vertical position to 0–1 volume (top = loud, bottom = quiet)
                motion_engine.current_volume = 1.0 - ny

                response['pointer'] = {
                    'x': float(index_tip.x),   # normalized 0–1
                    'y': float(index_tip.y),   # normalized 0–1
                    'hand': hand_label,        # "Left" or "Right"
                }
            # Track wrist positions for volume control
            # wrist = hand_landmarks.landmark[0]
            # if hand_label == "Left":
            #     left_wrist_y = float(wrist.y)
            # elif hand_label == "Right":
            #     right_wrist_y = float(wrist.y)
            
            response['hands'].append({
                'label': hand_label,
                'landmarks': landmarks,
                'finger_count': finger_count,
                'gestures': {
                    'ok_sign': is_ok,
                    # 'thumbs_up': is_thumbs_up,
                    # 'fist': is_fist_gesture,
                    'rock': is_rock,
                }
            })
            response['volume'] = float(motion_engine.current_volume)
    else:
        # No hands detected - reset current instrument but keep pending
        motion_engine.current_instrument = None
    
    # Volume control: based on average wrist height when both hands visible
    # if left_wrist_y is not None and right_wrist_y is not None:
    #     avg_y = (left_wrist_y + right_wrist_y) / 2.0
    #     # Invert: top of screen (y=0) = high volume, bottom (y=1) = low volume
    #     motion_engine.current_volume = 1.0 - avg_y
    #     # Clamp between 0 and 1
    #     motion_engine.current_volume = max(0.0, min(1.0, motion_engine.current_volume))
    
    # Play/Pause toggle: detect two fists
    # two_fists_now = (fist_count >= 2)
    # if two_fists_now and not motion_engine.prev_two_fists:
    #     # Rising edge - toggle play state
    #     motion_engine.is_playing = not motion_engine.is_playing
    #     response['play_toggle'] = True
    # motion_engine.prev_two_fists = two_fists_now
    
    # Shuffle trigger: detect rock gesture (rising edge)
    if rock_now and not motion_engine.prev_rock_any:
        # Rising edge - trigger shuffle
        response['shuffle_triggered'] = True
    motion_engine.prev_rock_any = rock_now
    
    response['volume'] = motion_engine.current_volume
    response['playing'] = motion_engine.is_playing
    
    # Process face for expression/key mode
    # if face_results.multi_face_landmarks:
    #     face_landmarks = face_results.multi_face_landmarks[0]
    #     mouth = get_mouth_features(face_landmarks)
        
    #     # Update neutral calibration
    #     motion_engine.update_neutral_curvature(mouth)
        
    #     if motion_engine.neutral_done:
    #         curv = mouth["curvature"]
    #         curv_delta = curv - motion_engine.neutral_curvature
    #         SMILE_DELTA = -0.01
            
    #         motion_engine.candidate_key = "major" if curv_delta < SMILE_DELTA else "minor"
            
    #         # Update key mode if any OK gesture is active
    #         any_ok = any(h.get('gestures', {}).get('ok_sign', False) for h in response['hands'])
    #         if any_ok and motion_engine.candidate_key != motion_engine.key_mode:
    #             motion_engine.key_mode = motion_engine.candidate_key
        
    #     opera_enabled = detect_opera(mouth)
        
    #     response['face'] = {
    #         'mouth_features': {
    #             'width': float(mouth['mouth_width']),
    #             'height': float(mouth['mouth_height']),
    #             'curvature': float(mouth['curvature']),
    #         },
    #         'opera_enabled': opera_enabled,
    #         'neutral_done': motion_engine.neutral_done,
    #     }
    
    # response['key_mode'] = motion_engine.key_mode
    # response['current_track'] = motion_engine.current_track
    # response['current_instrument'] = motion_engine.current_instrument
    # response['pitch'] = motion_engine.current_pitch
    
    return response

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Motion tracking server running'})

if __name__ == '__main__':
    print("🚀 Starting Motion Tracking Flask Server on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
