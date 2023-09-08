// layout.tsx
import React, { ReactNode } from 'react';

type LayoutProps = {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col justify-center items-center">
      {children}
    </div>
  );
}
