/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    serverComponentsExternalPackages: ['child_process']
  },
  webpack: (config, { isServer }) => {
    if (isServer) {
      // Add externals for server-side dependencies
      config.externals = config.externals || []
      config.externals.push('child_process')
    }
    return config
  }
}

module.exports = nextConfig
