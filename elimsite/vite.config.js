import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: "/elim/", // Essential for correct absolute asset paths on GH Pages
  server: {
    fs: {
      allow: [".."],
    },
  },
});
