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
import guideline from "./assets/guideline.png"
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
  const [currentTheme, setCurrentTheme] = useState("jazz");
  const [lastShuffleState, setLastShuffleState] = useState(false);
  const [shuffleCooldown, setShuffleCooldown] = useState(false);

  const themes = ["jazz", "house", "chill", "rock"];

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
    
    // Handle shuffle gesture (rock sign)
    if (data.shuffle_triggered && !lastShuffleState && !shuffleCooldown) {
      // Shuffle to next theme
      const currentIndex = themes.indexOf(currentTheme);
      const nextIndex = (currentIndex + 1) % themes.length;
      const nextTheme = themes[nextIndex];
      console.log(`🤘 Rock gesture detected! Shuffling from ${currentTheme} to ${nextTheme}`);
      handleThemeChange(nextTheme);
      
      // Set cooldown period
      setShuffleCooldown(true);
      setTimeout(() => {
        setShuffleCooldown(false);
        console.log('Shuffle cooldown ended');
      }, 2000); // 2 second cooldown
    }
    setLastShuffleState(data.shuffle_triggered || false);
    
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

  useEffect(() => {
    console.log(`BPM updated to: ${bpm}`);
    instrumentManager.setTempo(bpm);
  }, [bpm]);

  // Load all instruments on mount
  useEffect(() => {
    const instruments = ["keyboard", "guitar", "bass", "percussion"];
    themes.forEach((theme) => {
      instruments.forEach((inst) => {
        instrumentManager.loadInstrument(
          `${theme}_${inst}`, // unique ID
          `/assets/${theme}/${theme}_${inst}.mp3` // correct file path
        );
      });
    });
  }, []);

  // Handle theme change
  const handleThemeChange = (theme) => {
    console.log(`Changing theme from ${currentTheme} to ${theme}`);
    
    // Store which instruments are currently playing
    const currentlyPlaying = Object.keys(playingInstruments).filter(
      inst => playingInstruments[inst]
    );
    
    // Change theme immediately
    setCurrentTheme(theme);
    instrumentManager.setTheme(theme);
    
    // Restart the instruments that were playing instantly (no delay)
    currentlyPlaying.forEach(inst => {
      instrumentManager.toggle(inst);
    });
  };

  // Handle volume change
  const handleVolumeChange = (name, value) => {
    setVolumes((prev) => ({ ...prev, [name]: value }));
    instrumentManager.setVolume(name, value);
  };

return (
  <div className="App">
    {/* Header */}
    <div style={{ textAlign: "center", marginBottom: "20px" }}>
      <img 
        src={heartlogo} 
        alt="Heart Jam Logo" 
        style={{ width: "450px", marginTop: "5px" }} 
      />
      
      {/* Theme Buttons
      <div style={{ marginTop: "15px", marginBottom: "10px" }}>
        <button 
          onClick={() => handleThemeChange("jazz")}
          style={{ fontWeight: currentTheme === "jazz" ? "bold" : "normal" }}
        >
          Jazz
        </button>
        <button 
          onClick={() => handleThemeChange("house")}
          style={{ fontWeight: currentTheme === "house" ? "bold" : "normal" }}
        >
          House
        </button>
        <button 
          onClick={() => handleThemeChange("chill")}
          style={{ fontWeight: currentTheme === "chill" ? "bold" : "normal" }}
        >
          Chill
        </button>
        <button 
          onClick={() => handleThemeChange("rock")}
          style={{ fontWeight: currentTheme === "rock" ? "bold" : "normal" }}
        >
          Rock
        </button>
      </div> */}
      
      <FitbitConnector onBpmChange={setBpm} />
    </div>

    {/* Text and Theme Display Row */}
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0 130px", marginTop: "40px", marginBottom: "18px", maxWidth: "1400px", margin: "40px auto 18px auto" }}>
      {/* Bottom Left Text */}
      <div
        className="pixelify-sans"
        style={{
          fontSize: "48px",
          textAlign: "left",
          lineHeight: "1.2",
        }}
      >
        Listen to your heart...
      </div>

      {/* Current Theme Display */}
      <div 
        className="pixelify-sans"
        style={{
          background: currentTheme === "jazz" ? "linear-gradient(135deg, #556bc3ff 0%, #8352b4ff 100%)" :
                      currentTheme === "house" ? "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)" :
                      currentTheme === "chill" ? "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)" :
                      "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
          padding: "15px 30px",
          borderRadius: "20px",
          fontSize: "28px",
          fontWeight: "bold",
          color: "white",
          textShadow: "2px 2px 4px rgba(0,0,0,0.3)",
          boxShadow: "0 4px 15px rgba(0,0,0,0.2)",
          textTransform: "uppercase",
          letterSpacing: "2px",
          animation: "pulse 2s ease-in-out infinite",
          marginRight: "120px",
        }}
      >
        🎵 {currentTheme} Vibes 🎵
      </div>
    </div>

    {/* Main Content - Two Column Layout */}
    <div style={{ display: "flex", justifyContent: "center", alignItems: "flex-start", padding: "0 10px", gap: "0px", maxWidth: "1400px", margin: "0 auto" }}>
      
      {/* Left Side - Camera and Hearts */}
      <div style={{ flex: "1", display: "flex", flexDirection: "column", alignItems: "center", gap: "10px" }}>
        <CameraFeed onMotionData={handleMotionData} />
        
        {/* Pixel Hearts */}
        <div className="instruments-container" style={{ display: "flex", flexDirection: "row", justifyContent: "center", gap: "20px" }}>
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

        {/* Volume Controls */}
        <div style={{ width: "100%", maxWidth: "400px", position: "relative" }}>
          {["keyboard", "guitar", "percussion", "bass"].map((inst) => (
            <div key={inst} style={{ marginTop: "15px", display: "flex", alignItems: "center" }}>
              <span style={{ width: "100px", textTransform: "capitalize" }}>{inst}</span>
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

      {/* Right Side - Guidelines */}
      <div style={{ flex: "0 0 auto", display: "flex", alignItems: "flex-start", paddingRight: "120px" }}>
        <img
          src={guideline}
          alt="guideline"
          style={{
            width: "500px",
            height: "auto",
            display: "block",
          }}
        />
      </div>
    </div>
  </div>
  );
}
export default App;
