// src/components/Heart/Heart.jsx
import React, { useState, useRef } from "react";
import "./Heart.css";

// Standard Bluetooth SIG UUIDs for Heart Rate
const HEART_RATE_SERVICE_UUID = 0x180d;
const HEART_RATE_CHAR_UUID = 0x2a37;

export default function Heart() {
  const [bpm, setBpm] = useState(0);
  const [status, setStatus] = useState("Not connected");

  const deviceRef = useRef(null);
  const characteristicRef = useRef(null);

  const effectiveBpm = bpm > 0 ? bpm : 75;
  const duration = 60 / effectiveBpm + "s";

  const handleCharacteristicValueChanged = (event) => {
    const value = event.target.value;   // DataView
    const heartrate = value.getUint8(1); // simple parse (like your JS snippet)
    setBpm(heartrate);
  };

  const handleDisconnected = () => {
    setStatus("Disconnected");
    characteristicRef.current = null;
    deviceRef.current = null;
    // add: setBpm(0);
  };

  const connectHeartRate = async () => {
    if (!navigator.bluetooth) {
      alert("Web Bluetooth is not supported in this browser.");
      return;
    }

    setStatus("Requesting Bluetooth device…");

      navigator.bluetooth
      .requestDevice({ filters: [{ services: [HEART_RATE_SERVICE_UUID] }] })
      .then((device) => {
        setStatus("Connecting…");
        return device.gatt.connect();
      })
      .then((server) => server.getPrimaryService(HEART_RATE_SERVICE_UUID))
      .then((service) => service.getCharacteristic(HEART_RATE_CHAR_UUID))
      .then((characteristic) =>
        characteristic.startNotifications().then(() => characteristic)
      )
      .then((characteristic) => {
        characteristic.addEventListener(
          "characteristicvaluechanged",
          handleCharacteristicValueChanged
        );
        setStatus("Connected ✓");
      })
      .catch((error) => {
        console.error(error);
        setStatus("Error: " + error.message);
      });
  };

  return (
    <div className="heart-container">
      <div className="heart-controls">
        <button
          type="button"
          onClick={connectHeartRate}
          disabled={!navigator.bluetooth}
        >
          Connect Heart Rate Monitor
        </button>
        <div className="heart-status">{status}</div>
        <div className="heart-bpm-display">
          Current BPM: <strong>{bpm > 0 ? bpm : "--"}</strong>
        </div>
      </div>

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
    </div>
    
  );
}