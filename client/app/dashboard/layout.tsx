// layout.tsx
import React, { ReactNode } from 'react';

interface LayoutProps {
  children: ReactNode;
}

export default function Card(children : ReactNode){
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col justify-center items-center">
      {children}
    </div>
  );
}


