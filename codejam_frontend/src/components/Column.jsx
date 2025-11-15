import React from 'react';
import { SortableContext, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { Instrument } from './Instrument';
import './Column.css';

export const Column = ({ id, Instruments, maxSlots = 5 }) => {
  const slots = [...Instruments];
  while (slots.length < maxSlots) slots.push(null);

  return (
    <div className="column" data-column-id={id}>
      <SortableContext
        id={id}
        items={Instruments.map(i => i.id)}
        strategy={verticalListSortingStrategy}
      >
        {slots.map((instrument, index) =>
          instrument ? (
            <Instrument
              key={instrument.id}
              id={instrument.id}
              title={instrument.title}
              imgSrc={instrument.imgSrc}
              alt={instrument.alt}
            />
          ) : (
            <div
              key={`empty-${id}-${index}`}
              className="empty-slot"
              id={`empty-${id}-${index}`}
            />
          )
        )}
      </SortableContext>
    </div>
  );
};




