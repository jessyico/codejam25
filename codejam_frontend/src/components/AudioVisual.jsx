import React, { useEffect, useRef, useState } from "react";
import * as Tone from "tone";

export default function AudioVisualizer({ audioElement }) {
  const canvasRef = useRef(null);
  const audioCtxRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const analyserRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const synthRef = useRef(null);
  const oscillatorRef = useRef(null);

  // Initialize Tone.js synth for testing
  useEffect(() => {
    const synth = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: "triangle" },
      envelope: { attack: 0.005, decay: 0.1, sustain: 0.3, release: 1 },
    }).toDestination();
    synthRef.current = synth;

    return () => {
      synth.dispose();
    };
  }, []);

  // Play test notes (C major scale)
  const playTestNotes = async () => {
    await Tone.start();
    setIsPlaying(true);

    const notes = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"];
    const now = Tone.now();

    notes.forEach((note, index) => {
      synthRef.current.triggerAttackRelease(note, "0.5", now + index * 0.5);
    });

    setTimeout(() => {
      setIsPlaying(false);
    }, notes.length * 500 + 1000);
  };

  // Setup visualizer connected to Tone.js
  useEffect(() => {
    if (!canvasRef.current) return;

    // Get Tone.js destination (where all audio routes)
    const toneDestination = Tone.getDestination();

    // Create Web Audio Context connection
    if (!audioCtxRef.current) {
      audioCtxRef.current = Tone.getContext().rawContext;
    }
    const audioCtx = audioCtxRef.current;

    // Create analyser connected to Tone.js output
    if (!analyserRef.current) {
      analyserRef.current = audioCtx.createAnalyser();
      analyserRef.current.fftSize = 256;
      toneDestination.connect(analyserRef.current);
    }

    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let animationId;

    const draw = () => {
      animationId = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);

      const { width, height } = canvas;
      ctx.fillStyle = "#0b1020";
      ctx.fillRect(0, 0, width, height);

      const barWidth = (width / bufferLength) * 2.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * height;
        ctx.fillStyle = `hsl(${(i / bufferLength) * 360}, 100%, 50%)`;
        ctx.fillRect(x, height - barHeight, barWidth - 2, barHeight);
        x += barWidth;
      }
    };

    draw();

    return () => {
      cancelAnimationFrame(animationId);
    };
  }, []);

  return (
    <div>
      <button
        onClick={playTestNotes}
        disabled={isPlaying}
        style={{
          padding: "10px 20px",
          fontSize: "16px",
          cursor: isPlaying ? "not-allowed" : "pointer",
          opacity: isPlaying ? 0.6 : 1,
        }}
      >
        {isPlaying ? "Playing..." : "Play Test Notes"}
      </button>
      <canvas
        ref={canvasRef}
        width={500}
        height={150}
        style={{
          width: "100%",
          background: "#0b1020",
          borderRadius: 8,
          display: "block",
          marginTop: 16,
        }}
      />
    </div>
  );
}