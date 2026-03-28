import { useEffect, useState, useLayoutEffect, useRef } from "react";

type PopupState = "idle" | "recording" | "processing";

export function Popup() {
  const [state, setState] = useState<PopupState>("idle");
  const [amplitude, setAmplitude] = useState(0);
  const prefersReducedMotion = useRef(
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );

  useLayoutEffect(() => {
    document.documentElement.style.cssText =
      "background: transparent !important;";
    document.body.style.cssText =
      "background: transparent !important; margin: 0; padding: 0;";
    document.documentElement.classList.add("popup-transparent");

    const root = document.getElementById("root");
    if (root) {
      root.style.cssText = "background: transparent !important;";
    }
  }, []);

  useEffect(() => {
    const handleAmplitude = (e: CustomEvent<number>) => setAmplitude(e.detail);
    const handleState = (e: CustomEvent<{ state: PopupState }>) => {
      setState(e.detail.state);
    };

    document.addEventListener("amplitude", handleAmplitude as EventListener);
    document.addEventListener("popup-state", handleState as EventListener);

    return () => {
      document.removeEventListener("amplitude", handleAmplitude as EventListener);
      document.removeEventListener("popup-state", handleState as EventListener);
    };
  }, []);

  return (
    <div
      className="w-screen h-screen flex items-center justify-center select-none"
      style={{ background: "transparent" }}
    >
      {/* IDLE: Tiny pill */}
      {state === "idle" && (
        <div
          style={{
            width: "32px",
            height: "4px",
            borderRadius: "2px",
            background: "rgba(255, 255, 255, 0.15)",
          }}
        />
      )}

      {/* RECORDING: Simple bars */}
      {state === "recording" && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "3px",
            padding: "6px 10px",
            borderRadius: "12px",
            background: "rgba(0, 0, 0, 0.5)",
            backdropFilter: "blur(12px)",
          }}
        >
          {[0, 1, 2, 3, 4].map((i) => {
            const center = 2;
            const distance = Math.abs(i - center);
            const scale = 1 - distance * 0.2;
            const height = 4 + amplitude * 10 * scale;

            return (
              <div
                key={i}
                style={{
                  width: "2px",
                  height: `${height}px`,
                  borderRadius: "1px",
                  background: "rgba(52, 211, 153, 0.9)",
                  transition: "height 50ms ease-out",
                }}
              />
            );
          })}
        </div>
      )}

      {/* PROCESSING: Three dots */}
      {state === "processing" && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "4px",
            padding: "8px 12px",
            borderRadius: "12px",
            background: "rgba(0, 0, 0, 0.5)",
            backdropFilter: "blur(12px)",
          }}
        >
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              style={{
                width: "4px",
                height: "4px",
                borderRadius: "50%",
                background: "rgba(255, 255, 255, 0.7)",
                animation: prefersReducedMotion.current
                  ? "none"
                  : "fade 1s ease-in-out infinite",
                animationDelay: `${i * 0.2}s`,
              }}
            />
          ))}
        </div>
      )}

    </div>
  );
}
