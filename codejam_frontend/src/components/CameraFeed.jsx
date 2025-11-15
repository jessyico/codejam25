import { useEffect, useRef } from "react";
import "./CameraFeed.css";

export default function CameraFeed() {
  const videoRef = useRef(null);

  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" }
        });
        videoRef.current.srcObject = stream;
      } catch (error) {
        console.error("Camera error:", error);
      }
    }

    startCamera();
  }, []);

  return (
    <div className="camera-container">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="camera-video"
      />
    </div>
  );
}
