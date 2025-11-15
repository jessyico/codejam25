import React, { useState } from "react";
import "./Instrument.css";

export default function Instrument({ imgSrc, alt, onClick, showFire = false }) {
  const [fireVisible, setFireVisible] = useState(false);

  const handleClick = () => {
    if (showFire) {
      setFireVisible(true);
      setTimeout(() => setFireVisible(false), 1000);
    }

    if (onClick) onClick();
  };

  return (
    <div className="instrument" onClick={handleClick}>
      <img src={imgSrc} alt={alt} />

      {fireVisible && <span className="emoji-fire">🔥</span>}
    </div>
  );
}

