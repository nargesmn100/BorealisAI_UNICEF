import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Fine-Scale Poverty Estimation from Coarse Regional Labels',
  description: 'Interactive demonstration of weakly supervised spatial prediction and aggregation',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-white">{children}</body>
    </html>
  );
}
