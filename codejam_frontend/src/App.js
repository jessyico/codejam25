import { useState, useEffect } from "react";
import "./App.css";
import CameraFeed from "./components/CameraFeed";
import Heart from "./components/Heart";
import Instrument from "./components/Instrument";
import blueheart from "./assets/blue_heart.webp";
import yellowheart from "./assets/yellow_heart.webp";
import greenheart from "./assets/green_heart.webp";
import purpleheart from "./assets/purple_heart.webp";
import heartlogo from "./assets/heartjamlogo.png";
import FitbitConnector from "./components/FitbitConnect";
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
// Simulate incoming sensor data
// Sync instruments to BPM
<FitbitConnector onBpmChange={setBpm} />

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
    <div>
      <h1 style={{ color: "pink", fontFamily: "Barriecito" }}>
        Listen to your heart
      </h1>
      {/* 🌈 THEME BUTTONS ADDED HERE */}
      <div style={{ marginBottom: "20px" }}>
        <button onClick={() => instrumentManager.setTheme("jazz")}>Jazz</button>
        <button onClick={() => instrumentManager.setTheme("house")}>House</button>
        <button onClick={() => instrumentManager.setTheme("chill")}>Chill</button>
        <button onClick={() => instrumentManager.setTheme("rock")}>Rock</button>
      </div>
      
      <FitbitConnector onBpmChange={setBpm} />
      


      <div className="instruments-container">
        <Instrument
          imgSrc={blueheart}
          alt="blue heart"
          onClick={() => {
          instrumentManager.toggle('keyboard'); // template literal
          }}
        />
        <Instrument
          imgSrc={yellowheart}
          alt="yellow heart"
          onClick={() => {
          instrumentManager.toggle('guitar'); // template literal
          }}
        />
        <Instrument 
          imgSrc={greenheart} 
          alt="green heart" 
          onClick={() => {
          instrumentManager.toggle('percussion'); // template literal
          }}
        />
        <Instrument
          imgSrc={purpleheart}
          alt="purple heart"
          onClick={() => {
          instrumentManager.toggle('bass'); // template literal
          }}
        />
      </div>
<p>Current BPM: {bpm}</p>
<p>Current BPM: {bpm}</p>
      <CameraFeed />
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
  );
}

export default App;
