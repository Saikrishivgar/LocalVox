import "@testing-library/jest-dom/vitest";

if (typeof HTMLElement !== "undefined") {
  HTMLElement.prototype.scrollIntoView = () => {};
}

if (typeof HTMLCanvasElement !== "undefined") {
  HTMLCanvasElement.prototype.getContext = () => null;
}
