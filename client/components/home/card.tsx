// components/Card.tsx
import React, { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
}

export default function Card(children : ReactNode) {
  return (
    <div className="bg-white p-5 rounded-lg shadow-md">
      {children}
    </div>
  );
}