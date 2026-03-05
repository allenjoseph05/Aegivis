/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50:  "#eff8ff",
          100: "#dbeffe",
          200: "#b9e0fd",
          300: "#7cc9fc",
          400: "#00c4ff",
          500: "#0e8fea",
          600: "#0066e5",
          700: "#0052cc",
          800: "#003ab0",
          900: "#0d1f40",
          950: "#09162e",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
      },
    },
  },
  plugins: [],
};
