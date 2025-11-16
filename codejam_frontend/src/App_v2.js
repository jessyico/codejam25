import { useState, useEffect } from 'react';
import './App.css';
import Heart from "./components/Heart";
import AudioVisualizer from './components/AudioVisual';
import FitbitConnector from "./components/FitbitConnect";
import CameraFeed from './components/HandGestureCamera';

function App_v2() {
  const [bpm, setBpm] = useState(80); // default BPM
  const [motionData, setMotionData] = useState(null);
  const [currentTrack, setCurrentTrack] = useState(null);
  const [currentInstrument, setCurrentInstrument] = useState(null);

  // Handle motion data from camera
  const handleMotionData = (data) => {
    setMotionData(data);
    
    // Update app state from motion tracking
    if (data.current_track !== null) setCurrentTrack(data.current_track);
    if (data.current_instrument !== null) setCurrentInstrument(data.current_instrument);
  };


  return (
    <div>
      <h1 style={{ color: "pink", fontFamily: "Barriecito" }}>
      Listen to your heart
      </h1>

      <FitbitConnector onBpmChange={setBpm} />

<p>Current BPM: {bpm}</p>
      {/* Motion-tracked camera with gesture controls */}
      <CameraFeed onMotionData={handleMotionData} />
      
      {/* Display motion control state */}
      <div style={{ 
        background: 'rgba(0,0,0,0.8)', 
        color: 'white', 
        padding: '15px', 
        margin: '10px 0',
        borderRadius: '8px',
        fontFamily: 'monospace'
      }}>
        <h3>🎮 Motion Controls</h3>
        <p>Track: <strong>{currentTrack || 'None'}</strong> (right hand fingers + left OK)</p>
        <p>Instrument: <strong>{currentInstrument || 'None'}</strong> (left hand fingers + right OK)</p>
      </div>
      
      <Heart bpm={bpm} /> {/* instead of 80 would change to be currentbpm based on the sensor data  */} 
<h1>My Music Visual</h1>
    <AudioVisualizer />
  </div>
      
  );

}

export default App_v2;
