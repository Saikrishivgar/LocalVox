import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LocalVox — Offline AI Voice Companion",
  description:
    "Completely local Apple Silicon voice AI companion with persistent memory.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="font-sans antialiased selection:bg-sky-500/30 selection:text-sky-100">
        {children}
      </body>
    </html>
  );
}
