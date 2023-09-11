/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  images: {
    domains: ["lh3.googleusercontent.com", "vercel.com"],
  },
  experimental: {
    serverActions: true,
  },
  async redirects() {
    return [
      {
        source: "/github",
        destination: "https://github.com/abhayc-glitch",
        permanent: false,
      },
    ];
  },
};

module.exports = nextConfig;
