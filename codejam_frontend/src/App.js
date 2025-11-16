import { useState, useEffect } from "react";
import "./App.css";

import Heart from "./components/Heart";
import Instrument from "./components/Instrument";
import blueheart from "./assets/blue_heart.webp";
import yellowheart from "./assets/yellow_heart.webp";
import greenheart from "./assets/green_heart.webp";
import purpleheart from "./assets/purple_heart.webp";
import heartlogo from "./assets/heartjamlogo.png";
import FitbitConnector from "./components/FitbitConnect";
import drums from "./assets/drums_new.png";
import bass_new from "./assets/bass_new.png";
import guitar_new from "./assets/guitar_new.png";
import piano_new from "./assets/piano_new.png";
import CameraFeed from "./components/HandGestureCamera";
// import instrumentManager.js
import instrumentManager from "./audio/instrumentManager.js";

function App() {
  const [bpm, setBpm] = useState(80); // default BPM
  const [volumes, setVolumes] = useState({
    keyboard: 0,
    guitar: 0,
    bass: 0,
    percussion: 0,
  });
  const [motionData, setMotionData] = useState(null);
  const [currentTrack, setCurrentTrack] = useState(null);

  // Handle motion data from camera
  const handleMotionData = (data) => {
    setMotionData(data);
    // Update app state from motion tracking
    if (data.current_instrument !== null) setCurrentInstrument(data.current_instrument);
    
  };




// Simulate incoming sensor data
// Sync instruments to BPM
<FitbitConnector onBpmChange={setBpm} />
{/* Motion-tracked camera with gesture controls */}
<CameraFeed onMotionData={handleMotionData} />

useEffect(() => {
    console.log(`BPM updated to: ${bpm}`);
    instrumentManager.setTempo(bpm);
}, [bpm]);

  // Load all instruments on mount
  useEffect(() => {
    const instruments = ["keyboard", "guitar", "bass", "percussion"];
    const themes = ["jazz", "chill", "rock", "house"];
    themes.forEach((theme) => {
      instruments.forEach((inst) => {
        instrumentManager.loadInstrument(
          `${theme}_${inst}`, // unique ID
          `/assets/${theme}/${theme}_${inst}.mp3` // correct file path
        );
      });
    });
  }, []);

  // Handle volume change
  const handleVolumeChange = (name, value) => {
    setVolumes((prev) => ({ ...prev, [name]: value }));
    instrumentManager.setVolume(name, value);
  };

return (
  <div className="App"> {/* Start of App div */}
<div style={{ textAlign: "center" }}>
  <h1 style={{ color: "pink", fontFamily: "Barriecito" }}>
    
  </h1>
  <img 
    src={heartlogo} 
    alt="Heart Jam Logo" 
    style={{ width: "400px", height: "150px", marginTop: "10px" }} 
  />

      {/* 🌈 THEME BUTTONS ADDED HERE */}
      <div style={{ marginBottom: "20px" }}>
        <button onClick={() => instrumentManager.setTheme("jazz")}>Jazz</button>
        <button onClick={() => instrumentManager.setTheme("house")}>House</button>
        <button onClick={() => instrumentManager.setTheme("chill")}>Chill</button>
        <button onClick={() => instrumentManager.setTheme("rock")}>Rock</button>
      </div>
      <FitbitConnector onBpmChange={setBpm} />
      
      <CameraFeed />

      <div className="instruments-container" style={{ display: "flex", justifyContent: "flex-start", gap: "10px",paddingLeft: "200px" }}>
  {/* Left column */}
  <div className="column" style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
    <Instrument imgSrc={blueheart} alt="blue heart" onClick={() => instrumentManager.toggle('keyboard')} />
    <Instrument imgSrc={yellowheart} alt="yellow heart" onClick={() => instrumentManager.toggle('guitar')} />
    <Instrument imgSrc={greenheart} alt="percussion" onClick={() => instrumentManager.toggle('percussion')} />
    <Instrument imgSrc={purpleheart} alt="bass" onClick={() => instrumentManager.toggle('bass')} />
  </div>

  {/* Right column */}
  <div className="column" style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
    <Instrument imgSrc={piano_new} alt="piano icon" 
    style={{ height: "80px", width: "auto" }} 
    />
    <Instrument imgSrc={guitar_new} alt="guitar icon"
    style={{ height: "5px", width: "auto" }}  />
    <Instrument imgSrc={drums} alt="drums icon"
    style={{ height: "80px", width: "auto" }}  />
    <Instrument imgSrc={bass_new} alt="bass icon"
    style={{ height: "80px", width: "auto" }}  />
  </div>
</div>

    <Heart bpm={bpm} />
   
      {/* Volume control */}
      {["keyboard", "guitar", "percussion", "bass"].map((inst) => (
        <div key={inst} style={{ marginTop: "15px", display: "flex", alignItems: "center" }}>
          <span style={{ width: "40px", textTransform: "capitalize" }}>{inst}</span>
          <input
            type="range"
            min={-60}
           max={6}
            value={volumes[inst]}
            onChange={(e) =>
              handleVolumeChange(inst, parseFloat(e.target.value))
            }
            style={{ marginLeft: "10px", width: "200px" }}
          />
        </div>
      ))}
    </div>
    </div>
  );
}
export default App;
