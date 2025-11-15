import { useState, useEffect } from 'react';
import './App.css';
import { closestCorners, DndContext } from "@dnd-kit/core";
import { arrayMove } from "@dnd-kit/sortable";
import CameraFeed from './components/CameraFeed';
import Heart from './components/Heart';
import { Column } from './components/Column';
import catpianoGif from "./assets/piano.png";
import catguitarGif from "./assets/pixelguitar.png";
import drumGif from "./assets/pixeldrums.png";
import bassGif from "./assets/pixelbass.png";
import saxoGif from "./assets/saxophone.png";
import logo from "./assets/heartjamlogo.png";
import { SortableContext, verticalListSortingStrategy } from "@dnd-kit/sortable";

import React from 'react';

function App() {
  const [bpm, setBpm] = useState(80);
  const [Instruments, setInstruments] = useState([
    { id: 1, title: 'Piano', imgSrc: catpianoGif, alt: 'Funny cat piano' },
    { id: 2, title: 'Saxophone', imgSrc: saxoGif, alt: 'Saxophone' },
    { id: 3, title: 'Guitar', imgSrc: catguitarGif, alt: 'Funny cat guitar' },
    { id: 4, title: 'Drums', imgSrc: drumGif, alt: 'Fire drums' },
    { id: 5, title: 'Bass', imgSrc: bassGif, alt: 'Bass' },
  ]);
  const [BottomInstruments, setBottomInstruments] = useState([]);

  // Helper to find index in an array
  const getIndex = (arr, id) => arr.findIndex(item => item.id === id);

  const handleDragEnd = ({ active, over }) => {
    if (!over) return;
    if (active.id === over.id) return; // No move
  
    // Determine source array and setSource
    let sourceArray, setSource;
    if (getIndex(Instruments, active.id) !== -1) {
      sourceArray = Instruments;
      setSource = setInstruments;
    } else {
      sourceArray = BottomInstruments;
      setSource = setBottomInstruments;
    }
  
    // Determine destination array and setDest
    let destArray, setDest;
    if (over.id.toString().startsWith('empty')) {
      // Dropped on empty slot
      const [, colId] = over.id.toString().split('-');
      if (colId === 'bottom') {
        destArray = BottomInstruments;
        setDest = setBottomInstruments;
      } else if (colId === 'top') {
        destArray = Instruments;
        setDest = setInstruments;
      } else {
        return;
      }
    } else {
      // Dropped on an instrument
      if (getIndex(Instruments, over.id) !== -1) {
        destArray = Instruments;
        setDest = setInstruments;
      } else if (getIndex(BottomInstruments, over.id) !== -1) {
        destArray = BottomInstruments;
        setDest = setBottomInstruments;
      } else {
        return;
      }
    }
  
    // Get indices
    const oldIndex = getIndex(sourceArray, active.id);
    let newIndex;
    if (over.id.toString().startsWith('empty')) {
      // For empty slots, use the index from the id (e.g., empty-bottom-0)
      newIndex = parseInt(over.id.toString().split('-')[2]);
    } else {
      newIndex = getIndex(destArray, over.id);
    }
  
    // Move the item
    const item = sourceArray[oldIndex];
    setSource(prev => prev.filter(i => i.id !== active.id));
    setDest(prev => {
      const copy = [...prev];
      copy.splice(newIndex, 0, item);
      return copy;
    });
  };  

  useEffect(() => {
    const interval = setInterval(() => {
      setBpm(60 + Math.floor(Math.random() * 40));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
     <div style={{ display: "flex", justifyContent: "center", marginBottom: "20px" }}>
  <img src={logo} alt="HeartJam Logo" style={{ width: "500px" }} />
</div>

     {/*} <h1 className="pixelify-sans">Listen to your heart</h1>*/}
     <DndContext
  collisionDetection={closestCorners}
  onDragEnd={handleDragEnd}
>
  <div style={{ display: "flex", gap: "20px", alignItems: "flex-start" }}>
    {/* LEFT: Camera */}
    <div style={{ flex: 1 }}>
    { /*  <CameraFeed /> */}
    
      <p>Current BPM: {bpm}</p>
      <Heart bpm={bpm} />
      {/* RIGHT: Two columns */}
      <div style={{
        display: "flex",
        flexDirection: "row",
        width: "100%",
        gap: "20px",
        justifyContent: "space-between",
        alignItems: "flex-start"
      }}>
        <SortableContext
          items={[...Instruments, ...BottomInstruments].map(i => i.id)}
          strategy={verticalListSortingStrategy}
        >
          {/* First column centered 
          <div style={{ margin: "0 auto" }}>
            <Column id="bottom" Instruments={BottomInstruments} maxSlots={5} />
          </div>
          */}
          {/* Second column right-aligned */}
          <div style={{ margin: "0 auto" }}>
            <Column id="top" Instruments={Instruments} maxSlots={5} />
          </div>
        </SortableContext>
      </div>
    </div>
  </div>
</DndContext>


     
    </div>
  );
}

export default App;
