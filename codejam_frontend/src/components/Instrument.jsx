import React, { useState } from "react";
import "./Instrument.css";

export default function Instrument({ imgSrc, alt, onClick, showFire = false, isPlaying = false, beat = false, instrumentName = "" }) {
  const [fireVisible, setFireVisible] = useState(false);

  const handleClick = () => {
    if (showFire) {
      setFireVisible(true);
      setTimeout(() => setFireVisible(false), 1000);
    }

    if (onClick) onClick();
  };

  return (
    <div className="instrument" onClick={handleClick} title={instrumentName}>
      <img 
        src={imgSrc} 
        alt={alt} 
        className={beat && isPlaying ? "heartbeat" : ""}
        style={{
          filter: isPlaying ? 'brightness(1) saturate(1.2)' : 'brightness(0.8) saturate(0.8)',
          transition: 'filter 0.3s ease'
        }}
      />
      {fireVisible && <span className="emoji-fire">🔥</span>}
    </div>
  );
}