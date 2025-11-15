import { useState, useEffect } from 'react';
import './App.css';
import CameraFeed from './components/CameraFeed';
import Heart from "./components/Heart";

function App() {
  const [bpm, setBpm] = useState(80); // default BPM

  // Simulate incoming sensor data
  useEffect(() => {
    // Example: every 2 seconds, update BPM randomly
    const interval = setInterval(() => {
      const newBpm = 60 + Math.floor(Math.random() * 40); // 60-100 BPM
      setBpm(newBpm);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <h1 style={{ color: "pink", fontFamily: "Barriecito" }}>
  Listen to your heart
</h1>
<p>Current BPM: {bpm}</p>
      <CameraFeed />
      <Heart bpm={80} /> {/* instead of 80 would change to be currentbpm based on the sensor data  */} 
      
    </div>
  );
}

export default App;
