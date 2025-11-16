// src/components/FitbitConnector.jsx
import React, { useEffect, useState } from "react";
import "./FitbitConnect.css";

const FITBIT_CLIENT_ID = "23TPXV";      // <- put your client id
const REDIRECT_URI = "http://localhost:3000/";       // must match Fitbit app
const SCOPES = "heartrate";                          // add more if you want

function FitbitConnector({ onBpmChange }) {
  const [accessToken, setAccessToken] = useState(null);
  const [status, setStatus] = useState("Not connected");
  const [bpm, setBpm] = useState(null);
  const [rawJson, setRawJson] = useState(null); // for debugging

  // 1) On load, see if Fitbit redirected us back with a token
  useEffect(() => {
    if (window.location.hash.startsWith("#")) {
      const params = new URLSearchParams(window.location.hash.substring(1));
      const token = params.get("access_token");

      if (token) {
        console.log("[Fitbit] Got access token:", token);
        setAccessToken(token);
        setStatus("Connected");

        sessionStorage.setItem("fitbit_access_token", token);
        // Clean the hash so we don't keep re-parsing
        window.history.replaceState({}, document.title, window.location.pathname);
      }
    } else {
      const saved = sessionStorage.getItem("fitbit_access_token");
      if (saved) {
        console.log("[Fitbit] Restored token from sessionStorage");
        setAccessToken(saved);
        setStatus("Connected");
      }
    }
  }, []);

  // 2) Build Fitbit auth URL and redirect there
  const handleConnectClick = () => {
    const authUrl = new URL("https://www.fitbit.com/oauth2/authorize");
    authUrl.searchParams.set("response_type", "token"); // implicit grant
    authUrl.searchParams.set("client_id", FITBIT_CLIENT_ID);
    authUrl.searchParams.set("redirect_uri", REDIRECT_URI);
    authUrl.searchParams.set("scope", SCOPES);
    authUrl.searchParams.set("expires_in", "31536000"); // 1 year

    window.location.href = authUrl.toString();
  };

  // 3) Fetch today's heart data (summary + try intraday)
  const fetchHeartRate = async () => {
    if (!accessToken) {
      setStatus("Not connected");
      return;
    }

    try {
      setStatus("Fetching...");

      const now = new Date();
      const year = now.getFullYear();
      const month = String(now.getMonth() + 1).padStart(2, '0');
      const day = String(now.getDate()).padStart(2, '0');
      const today = `${year}-${month}-${day}`; // YYYY-MM-DD in local time
      const url = `https://api.fitbit.com/1/user/-/activities/heart/date/${today}/1d/1min.json`;

      console.log("[Fitbit] Fetching:", url);

      const res = await fetch(url, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });

      console.log("[Fitbit] HTTP status:", res.status);

      if (!res.ok) {
        const text = await res.text();
        console.error("[Fitbit] Error body:", text);
        throw new Error(`Fitbit API error: ${res.status}`);
      }

      const data = await res.json();
      console.log("[Fitbit] Response JSON:", data);
      setRawJson(data); // so you can inspect it on screen while debugging

      let currentBpm = null;

      // Try intraday first
      const intraday = data["activities-heart-intraday"];
      if (intraday && intraday.dataset && intraday.dataset.length > 0) {
        const lastSample = intraday.dataset[intraday.dataset.length - 1];
        currentBpm = lastSample.value;
        console.log("[Fitbit] Using intraday BPM:", currentBpm);
      } else {
        // Fallback: use summary resting heart rate if intraday isn’t available
        const summary = data["activities-heart"]?.[0];
        const resting = summary?.value?.restingHeartRate;
        if (resting) {
          currentBpm = resting;
          console.log("[Fitbit] Using restingHeartRate:", currentBpm);
        } else {
          console.warn("[Fitbit] No intraday or resting HR available");
        }
      }

      if (currentBpm != null) {
        setBpm(currentBpm);
        setStatus("Updated");
        if (onBpmChange) onBpmChange(currentBpm);
      } else {
        setStatus("No heart-rate data available for today");
      }
    } catch (err) {
      console.error(err);
      setStatus(`Error: ${err.message}`);
    }
  };

  // Auto-fetch once when we first get a token
  useEffect(() => {
    if (accessToken) {
      fetchHeartRate();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken]);

   useEffect(() => {
    if (!accessToken) return;
    const interval = setInterval(() => {
       fetchHeartRate();
     }, 60000); // 10,000 ms = 10 seconds

     return () => clearInterval(interval);
   }, [accessToken]);

  return (
    <div style={{ marginBottom: "1rem" }}>
     <button
        type="button"
        className="connect-sensor-btn"
        onClick={handleConnectClick}
        disabled={status === "Fetching..."}
      >
        Connect Sensor
        {status === "Fetching..." && <span className="spinner" />}
      </button>

      <div style={{ marginTop: "0.5rem" }}>
        <div><strong>Status:</strong> {status}</div>
        <div><strong>BPM:</strong> {bpm ?? "--"}</div>
      </div>

      {/* Optional: show a small snippet of raw JSON for debugging */}
      {rawJson && (
        <pre style={{ maxHeight: 200, overflow: "auto", fontSize: 10 }}>
            {JSON.stringify(rawJson, null, 2)}
        </pre>
      )}
    </div>
  );
}

export default FitbitConnector;
