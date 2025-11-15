// src/components/Heart/Heart.jsx
import React from "react";
import "./Heart.css";

export default function Heart({ bpm = 75 }) {
  const duration = 60 / bpm + "s"; // seconds per beat

  return (
    <div className="ekg-wrapper">
      <svg
        key={bpm}  // <--- this forces React to recreate the element
        className="ekg-line"
        viewBox="0 0 500 100"
        preserveAspectRatio="none"
        style={{ animationDuration: duration }}
>

        <path
          d="M0 50 L100 50 L120 30 L140 70 L160 50 L250 50 
             L270 20 L290 80 L310 50 L500 50"
          fill="none"
          stroke="pink"
          strokeWidth="4"
        />
      </svg>

      {/* Red heart in the center of the EKG */}
      <div
        className="heart-icon"
        style={{ animationDuration: duration }}
      >
        ❤️
      </div>
    </div>
  );
}

