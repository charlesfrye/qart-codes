/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        degular: ["Degular", "sans-serif", "system-ui"],
        inter: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        pixel: ["Pixeloid", "monospace", "sans-serif", "system-ui"],
      },
      colors: {
        black: `#000000`,
        green: `#84CC16`,
        "green-light": `#DDFFDC`,
        "green-bright": `#7FEE64`,
        gray: "#161916",
      },
    },
  },
  plugins: [],
};
