import React, { useState } from "react";
import "./Instrument.css";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";

export const Instrument = ({ id, title, imgSrc, alt }) => {
  const { attributes, listeners, setNodeRef, transform, transition } = useSortable({ id });
  const [isZoomed, setIsZoomed] = useState(false);

  const style = {
    transition,
    transform: CSS.Transform.toString(transform),
  };

  const imgStyle = {
    width: "100px",
    transform: isZoomed ? "scale(1.5)" : "scale(1)",
    transition: "transform 0.2s ease",
    cursor: "pointer",
  };

  const handlePointerDown = (e) => {
    // Only toggle zoom on left click, not during drag
    if (e.button === 0 && !e.ctrlKey && !e.shiftKey) {
      e.preventDefault();
      setIsZoomed(!isZoomed);
    }
  };

  return (
    <div
      ref={setNodeRef}
      {...attributes}
      {...listeners}
      style={style}
      className="instrument"
    >
      <img
        src={imgSrc}
        alt={alt}
        style={imgStyle}
        onPointerDown={handlePointerDown}
      />
      <div>{title || alt}</div>
    </div>
  );
};
