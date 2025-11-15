
import React from "react";
import mouseImg from "../assets/xp_cursor.png";

import "./PixelMouse.css"; // optional, for styles

export const PixelMouse = () => {
  return (
    <img 
      src={mouseImg} 
      alt="Retro pixel mouse" 
      className="pixel-mouse" 
    />
  );
};