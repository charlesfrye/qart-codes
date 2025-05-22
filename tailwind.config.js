/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        degular: ["Degular", "ui-sans-serif", "system-ui"],
      },
      colors: {
        black: `#18181B`,
        green: `#84CC16`,
        "green-light": `#DDFFDC`,
      },
    },
  },
  plugins: [],
};
