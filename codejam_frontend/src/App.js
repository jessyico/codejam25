import { useState, useEffect } from "react";
import "./App.css";

// import Heart from "./components/Heart";
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
  const [currentInstrument, setCurrentInstrument] = useState(null);
  const [instrumentCounters, setInstrumentCounters] = useState({
    1: 0, // keyboard
    2: 0, // guitar
    3: 0, // percussion
    4: 0, // bass
  });
  const [playingInstruments, setPlayingInstruments] = useState({
    keyboard: false,
    guitar: false,
    bass: false,
    percussion: false,
  });
  const [globalBeat, setGlobalBeat] = useState(false);

  // Global synchronized heartbeat
  useEffect(() => {
    const beatDuration = (60 / bpm) * 1000; // Convert BPM to milliseconds

    const interval = setInterval(() => {
      setGlobalBeat(true);
      setTimeout(() => setGlobalBeat(false), 150); // Quick pulse
    }, beatDuration);

    return () => clearInterval(interval);
  }, [bpm]);

  // Handle motion data from camera
  const handleMotionData = (data) => {
    setMotionData(data);
    
    console.log('Motion data received:', data);
    console.log('Current instrument from backend:', data.current_instrument);
    console.log('Instrument counters:', instrumentCounters);
    
    // Update app state from motion tracking
    if (data.current_instrument !== null && data.current_instrument !== currentInstrument) {
      const instrumentNum = data.current_instrument;
      console.log(`Instrument ${instrumentNum} selected`);
      setCurrentInstrument(instrumentNum);

      // Map instrument number to name
      const instrumentMap = {
        1: 'keyboard',
        2: 'guitar',
        3: 'percussion',
        4: 'bass'
      };
      
      const instrumentName = instrumentMap[instrumentNum];
      const currentCount = instrumentCounters[instrumentNum];
      const newCount = currentCount + 1;
      
      console.log(`Counter for instrument ${instrumentNum}: ${currentCount} -> ${newCount}`);
      
      // Toggle the instrument
      toggleInstrument(instrumentName);
      
      // Update counter - reset to 0 if it reaches 2
      setInstrumentCounters(prev => ({
        ...prev,
        [instrumentNum]: newCount >= 2 ? 0 : newCount
      }));
    }
    
    // Reset current instrument when gesture is released
    if (data.current_instrument === null) {
      setCurrentInstrument(null);
    }
  };

  // Toggle instrument and update playing state
  const toggleInstrument = (name) => {
    instrumentManager.toggle(name);
    setPlayingInstruments(prev => ({
      ...prev,
      [name]: !prev[name]
    }));
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
    style={{ width: "450px", marginTop: "10px" }} 
  />

      {/* 🌈 THEME BUTTONS ADDED HERE */}
      <div style={{ marginBottom: "20px" }}>
        <button onClick={() => instrumentManager.setTheme("jazz")}>Jazz</button>
        <button onClick={() => instrumentManager.setTheme("house")}>House</button>
        <button onClick={() => instrumentManager.setTheme("chill")}>Chill</button>
        <button onClick={() => instrumentManager.setTheme("rock")}>Rock</button>
      </div>
      <FitbitConnector onBpmChange={setBpm} />
      
      <CameraFeed onMotionData={handleMotionData} />

      <div className="instruments-container" style={{ display: "flex", flexDirection: "row", justifyContent: "center", gap: "20px", marginTop: "20px" }}>
    <Instrument 
      imgSrc={blueheart} 
      alt="blue heart" 
      onClick={() => toggleInstrument('keyboard')} 
      isPlaying={playingInstruments.keyboard}
      beat={globalBeat}
      instrumentName="Keyboard"
    />
    <Instrument 
      imgSrc={yellowheart} 
      alt="yellow heart" 
      onClick={() => toggleInstrument('guitar')} 
      isPlaying={playingInstruments.guitar}
      beat={globalBeat}
      instrumentName="Guitar"
    />
    <Instrument 
      imgSrc={greenheart} 
      alt="percussion" 
      onClick={() => toggleInstrument('percussion')} 
      isPlaying={playingInstruments.percussion}
      beat={globalBeat}
      instrumentName="Percussion"
    />
    <Instrument 
      imgSrc={purpleheart} 
      alt="bass" 
      onClick={() => toggleInstrument('bass')} 
      isPlaying={playingInstruments.bass}
      beat={globalBeat}
      instrumentName="Bass"
    />
  </div>

    {/* <Heart bpm={bpm} /> */}
   
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
