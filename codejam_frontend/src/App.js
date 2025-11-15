import { useState, useEffect } from 'react';
import './App.css';
import CameraFeed from './components/CameraFeed';
import Heart from "./components/Heart";
import catpianoGif from "./assets/cat-playing-piano-funny.gif"; // import the GIF
import catguitarGif from "./assets/cat-guitar.gif"; // import the GIF
import drumGif from "./assets/drums.gif"; // import the GIF
import bassGif from "./assets/bass-kitty.gif"; // import the GIF
import saxoGif from "./assets/saxo.gif"; // import the GIF
import Instrument from './components/Instrument';

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

<div className="instruments-container">
<Instrument
imgSrc={catpianoGif}
alt="Funny cat piano"
onClick={()=> alert("Meow! You played the cat piano!")}
/>
<Instrument
imgSrc={saxoGif}
alt="Saxophone"
onClick={()=> alert("Mr. Saxobeat")}
/>
<Instrument
imgSrc={catguitarGif}
alt="Funny cat guitar"
onClick={()=> alert("Meow! You played the cat guitar!")}
/>
<Instrument
imgSrc={drumGif}
alt="Fire drums"
showFire={true}
/>
<Instrument
imgSrc={bassGif}
alt="Bass"
onClick={()=> alert("You're all about that bass! (no treble here)")}
/>
</div>


<p>Current BPM: {bpm}</p>
      <CameraFeed />
      <Heart bpm={80} /> {/* instead of 80 would change to be currentbpm based on the sensor data  */} 
      
    </div>
  );
}

export default App;
