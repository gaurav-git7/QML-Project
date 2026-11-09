import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// dev: `npm run dev` proxies /api to the Python server on :8000; prod: `npm run build` -> dist/ (served by the Python server)
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:8000' } },
  build: { outDir: 'dist' },
})
