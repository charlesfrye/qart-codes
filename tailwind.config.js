/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        black: `#18181B`,
        green: `#84CC16`,
        "green-light": `#7FEE64`,
      },
    },
  },
  plugins: [],
};
