/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        blue: `rgb(64, 81, 181)`,
        orange: `rgb(246, 140, 8)`,
        "orange-light": `rgb(255, 145, 0)`,
      },
    },
  },
  plugins: [],
};
