import React from "react";
import {createRoot} from "react-dom/client";
import SplashCursor from "./SplashCursor.jsx";

export function mountSplash(container) {
    const root = createRoot(container);
    root.render(<SplashCursor colors={["#2246ef", "#5632ee"]} rainbow={false} theme="light"
        intensity={1.35} size={.68} force={1.02} swirl={3.6} trail={1.08}
        momentum={.78} shading={.28} glow={0} quality="low" contained
        intro={false} clickSplash zIndex={0} />);
    return () => root.unmount();
}
