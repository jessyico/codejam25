import { useEffect, useRef, useState } from "react";
import "./HandGestureCamera.css";

export default function CameraFeed({ onMotionData }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [motionData, setMotionData] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const animationFrameRef = useRef(null);

  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: 640, height: 480 }
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          // Wait for video to be ready before starting render loop
          await videoRef.current.play();
        }
      } catch (error) {
        console.error("Camera error:", error);
      }
    }
    startCamera();

    return () => {
      // Cleanup camera stream
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Separate animation loop for smooth rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    let isActive = true;

    const renderLoop = () => {
      if (!isActive) return;

      if (video.readyState >= video.HAVE_CURRENT_DATA) {
        // Set canvas size on first frame
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        // Draw video frame
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Draw overlay if we have motion data
        if (motionData) {
          drawOverlay(ctx, motionData, canvas.width, canvas.height);
        }
      }

      animationFrameRef.current = requestAnimationFrame(renderLoop);
    };

    // Start rendering once video starts playing
    video.addEventListener('playing', renderLoop);
    
    // Or start immediately if already playing
    if (video.readyState >= video.HAVE_CURRENT_DATA) {
      renderLoop();
    }

    return () => {
      isActive = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      video.removeEventListener('playing', renderLoop);
    };
  }, [motionData]);

  // Process frames and send to backend (separate from rendering)
  useEffect(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const interval = setInterval(async () => {
      if (isProcessing || videoRef.current.readyState !== 4) return;

      try {
        setIsProcessing(true);

        const video = videoRef.current;
        
        // Create a temporary canvas for capturing the frame
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(video, 0, 0);

        // Convert to base64 at lower quality for speed
        const frameData = tempCanvas.toDataURL('image/jpeg', 0.5);

        // Send to backend
        const response = await fetch('http://localhost:5001/api/motion/process-frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ frame: frameData })
        });

        if (response.ok) {
          const data = await response.json();
          setMotionData(data);
          
          // Pass data to parent component
          if (onMotionData) {
            onMotionData(data);
          }
        } else {
          console.error('Backend error:', response.status);
        }
      } catch (error) {
        console.error('Motion processing error:', error);
      } finally {
        setIsProcessing(false);
      }
    }, 20); // Process ~10 frames per second

    return () => clearInterval(interval);
  }, [onMotionData, isProcessing]);

  const drawOverlay = (ctx, data, width, height) => {
    // Save context and flip horizontally to match mirrored display
    ctx.save();
    ctx.scale(-1, 1); // Flip horizontally
    ctx.translate(-width, 0); // Move back into view
    
    // Draw hand landmarks
    if (data.hands && data.hands.length > 0) {
      data.hands.forEach(hand => {
        ctx.fillStyle = hand.label === 'Left' ? '#ed78d2ff' : '#61df9cff';
        ctx.strokeStyle = hand.label === 'Left' ? '#ed78d2ff' : '#61df9cff';
        ctx.lineWidth = 2;

        // Draw points
        hand.landmarks.forEach(lm => {
          // Flip x to match mirrored canvas display
          const x = (1 - lm.x) * width;
          const y = lm.y * height;
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, 2 * Math.PI);
          ctx.fill();
        });

        // Draw hand connections
        const connections = [
          [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
          [0, 5], [5, 6], [6, 7], [7, 8], // Index
          [0, 9], [9, 10], [10, 11], [11, 12], // Middle
          [0, 13], [13, 14], [14, 15], [15, 16], // Ring
          [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
          [5, 9], [9, 13], [13, 17], // Palm
        ];

        connections.forEach(([i, j]) => {
          const p1 = hand.landmarks[i];
          const p2 = hand.landmarks[j];
          ctx.beginPath();
          // Flip x to match mirrored canvas display
          ctx.moveTo((1 - p1.x) * width, p1.y * height);
          ctx.lineTo((1 - p2.x) * width, p2.y * height);
          ctx.stroke();
        });

        // Draw label
        const wrist = hand.landmarks[0];
        ctx.fillStyle = 'white';
        ctx.font = 'bold 16px Arial';
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 3;
        const label = `(${hand.finger_count} fingers)`;
        // Flip x to match mirrored canvas display
        ctx.strokeText(label, (1 - wrist.x) * width - 30, wrist.y * height - 10);
        ctx.fillText(label, (1 - wrist.x) * width - 30, wrist.y * height - 10);

        // Show gestures
        if (hand.gestures.ok_sign) {
          ctx.strokeText('✓ OK!', (1 - wrist.x) * width - 20, wrist.y * height + 30);
          ctx.fillText('✓ OK!', (1 - wrist.x) * width - 20, wrist.y * height + 30);
        }
      });
    }

    // Draw info overlay
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(10, 10, 250, 50);
    ctx.fillStyle = 'white';
    ctx.font = '14px monospace';
    ctx.fillText(`Instrument: ${data.current_instrument || 'none'}`, 20, 30);
    if (data.face && data.face.opera_enabled) {
      ctx.fillText('🎵 OPERA MODE ACTIVE', 20, 110);
    }
    if (data.face && !data.face.neutral_done) {
      ctx.fillText('⚠️ Calibrating...', 20, 130);
    }
    
    // Restore context to undo flip
    ctx.restore();
  };

  return (
    <div className="camera-container" style={{ position: 'relative' }}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ display: 'none' }}
      />
      <canvas
        ref={canvasRef}
        className="camera-video"
        style={{ maxWidth: '100%', height: 'auto', display: 'block' }}
      />
      {motionData && (
        <div style={{
          position: 'absolute',
          bottom: 10,
          right: 10,
          background: 'rgba(0,0,0,0.7)',
          color: 'white',
          padding: '5px 10px',
          borderRadius: '5px',
          fontSize: '12px'
        }}>
          {motionData.hands?.length || 0} hand(s) detected
        </div>
      )}
    </div>
  );
}