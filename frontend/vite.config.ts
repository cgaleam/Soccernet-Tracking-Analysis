import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  // Rutas relativas: el build resultante debe poder servirse desde
  // cualquier ruta/hosting estático sin configuración adicional (RNF-007).
  base: './',
  plugins: [react()],
})
