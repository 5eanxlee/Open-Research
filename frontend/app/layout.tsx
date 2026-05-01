import type { Metadata } from "next";
import localFont from "next/font/local";

import { AppProviders } from "@/components/app-providers";

import "katex/dist/katex.min.css";
import "./globals.css";

const sohne = localFont({
  src: [
    {
      path: "../public/fonts/TestSohne-Leicht.otf",
      weight: "300",
      style: "normal",
    },
    {
      path: "../public/fonts/TestSohne-Buch.otf",
      weight: "400",
      style: "normal",
    },
    {
      path: "../public/fonts/TestSohne-Kraftig.otf",
      weight: "700",
      style: "normal",
    },
  ],
  variable: "--font-testsohne",
});

export const metadata: Metadata = {
  title: "Open Research Console",
  description: "Operational dashboard for the Open Research deep research agent.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={sohne.variable}>
        <AppProviders>{children}</AppProviders>
      </body>
    </html>
  );
}
