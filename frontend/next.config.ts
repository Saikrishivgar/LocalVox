import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  transpilePackages: ["three"],
  outputFileTracingRoot: path.resolve(__dirname),
};

export default nextConfig;
