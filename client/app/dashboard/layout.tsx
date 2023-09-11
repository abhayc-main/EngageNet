// dashboard/layout.tsx
import React, { ReactNode } from 'react';

type LayoutProps = {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-cyan-100 p-10">
      <header className="mb-10">
        <h1 className="text-3xl font-bold">Demo</h1>
      </header>
      <main>
        {children}
      </main>
    </div>
  );
}
