import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  ...(process.env.NODE_ENV === "production" ? { output: "standalone" } : {}),
};

export default nextConfig;
