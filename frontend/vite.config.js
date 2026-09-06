import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API has no CORS middleware. Proxying in dev keeps a frontend-only concern
// out of api.py rather than adding middleware for it.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
