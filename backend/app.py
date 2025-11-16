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
    
    # Process hands
    hands_results = motion_engine.hands.process(frame_rgb)
    #face_results = motion_engine.face_mesh.process(frame_rgb)
    
    response = {
        'hands': [],
        'gestures': {},
        'current_instrument': getattr(motion_engine, 'current_instrument', None),
        'prev_rock_any': motion_engine.prev_rock_any,
        'prev_two_fists': motion_engine.prev_two_fists,
    }
    
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
            is_thumbs_up = is_thumb_up(hand_landmarks)
            is_fist_gesture = is_fist(hand_landmarks)
            is_rock = is_rock_sign(hand_landmarks)
            
            # Update pending selections based on hand
            if hand_label == "Left" and 4>= finger_count > 0:
                motion_engine.pending_instrument = finger_count
            
            # Confirm with OK gesture (right hand confirms left hand instrument selection)
            if hand_label == "Right" and is_ok:
                if motion_engine.pending_instrument is not None:
                    motion_engine.current_instrument = motion_engine.pending_instrument
            
            # Update pitch from pointer (1 finger)
            if finger_count == 1:
                index_tip = hand_landmarks.landmark[8]
                ny = float(index_tip.y)
                motion_engine.current_pitch = 1.0 - ny
            
            response['hands'].append({
                'label': hand_label,
                'landmarks': landmarks,
                'finger_count': finger_count,
                'gestures': {
                    'ok_sign': is_ok,
                    'thumbs_up': is_thumbs_up,
                    'fist': is_fist_gesture,
                    'rock': is_rock,
                }
            })
    
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
