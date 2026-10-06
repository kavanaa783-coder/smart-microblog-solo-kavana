import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// Backend runs on FastAPI (uvicorn) — default dev port 8000.
// Change VITE_API_BASE_URL in .env if your backend runs elsewhere.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  return {
    plugins: [react()],
    server: {
      host: '0.0.0.0',
      port: 5173,
      proxy: {
        '/api': {
          target: env.VITE_BACKEND_PROXY_TARGET || 'http://127.0.0.1:8000',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
  };
});
