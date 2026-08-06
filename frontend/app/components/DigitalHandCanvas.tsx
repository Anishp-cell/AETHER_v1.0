"use client";
import { useEffect, useRef } from "react";
import { GestureData } from "../hooks/useHandGesture";

interface Props {
  gestureData: GestureData;
}

export default function DigitalHandCanvas({ gestureData }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas with dark futuristic grid background
    ctx.fillStyle = "rgba(5, 10, 20, 0.95)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw subtle glowing cyber grid
    ctx.strokeStyle = "rgba(0, 245, 255, 0.05)";
    ctx.lineWidth = 1;
    const gridSize = 20;
    for (let x = 0; x < canvas.width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],     // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8],     // Index
      [5, 9], [9, 10], [10, 11], [11, 12], // Middle
      [9, 13], [13, 14], [14, 15], [15, 16], // Ring
      [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
      [0, 17],
    ];

    const drawHand = (handLandmarks: Array<{ x: number; y: number; z: number }>, color: string, glowColor: string) => {
      const w = canvas.width;
      const h = canvas.height;

      // Skeleton lines
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.shadowColor = glowColor;
      ctx.shadowBlur = 10;

      connections.forEach(([i, j]) => {
        const p1 = handLandmarks[i];
        const p2 = handLandmarks[j];
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1.x * w, p1.y * h);
          ctx.lineTo(p2.x * w, p2.y * h);
          ctx.stroke();
        }
      });

      // Joint nodes
      handLandmarks.forEach((pt, idx) => {
        const isTip = idx === 4 || idx === 8;
        ctx.fillStyle = isTip ? "#FFFFFF" : color;
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = isTip ? 14 : 6;

        ctx.beginPath();
        ctx.arc(pt.x * w, pt.y * h, isTip ? 4.5 : 2.5, 0, Math.PI * 2);
        ctx.fill();
      });
    };

    if (gestureData.landmarks && gestureData.landmarks.length === 21) {
      // Primary Hand (Cyan)
      drawHand(gestureData.landmarks, "#00F5FF", "#00F5FF");

      // Secondary Hand (Magenta) if present
      if (gestureData.secondLandmarks && gestureData.secondLandmarks.length === 21) {
        drawHand(gestureData.secondLandmarks, "#FF007F", "#FF007F");

        // Distance line between 2 hands
        const h1 = gestureData.landmarks[0];
        const h2 = gestureData.secondLandmarks[0];
        ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(h1.x * canvas.width, h1.y * canvas.height);
        ctx.lineTo(h2.x * canvas.width, h2.y * canvas.height);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    } else {
      // Searching text
      ctx.fillStyle = "rgba(0, 245, 255, 0.4)";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText("WAVE HAND(S) IN FRONT OF CAMERA", canvas.width / 2, canvas.height / 2);
    }
  }, [gestureData]);

  return (
    <div className="relative w-full h-full border border-cyan-500/30 rounded-lg overflow-hidden bg-black/80 flex flex-col">
      {/* Header Bar */}
      <div className="px-3 py-1.5 bg-black/60 border-b border-cyan-500/20 flex items-center justify-between z-10">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${gestureData.isTracking ? "bg-cyan-400 animate-ping" : "bg-yellow-400"}`} />
          <span className="text-[10px] font-mono text-cyan-300 tracking-wider uppercase">
            {gestureData.handCount > 1 ? "DUAL HAND TRACKING" : "DIGITAL HAND INTERFACE"}
          </span>
        </div>
        <span className="text-[9px] font-mono text-cyan-400/70">
          {gestureData.isTracking ? `${gestureData.handCount} HAND${gestureData.handCount > 1 ? "S" : ""} ACTIVE` : "SEARCHING"}
        </span>
      </div>

      {/* Canvas */}
      <canvas ref={canvasRef} width={240} height={180} className="w-full h-full object-cover" />

      {/* Footer Telemetry Overlay */}
      {gestureData.isTracking && (
        <div className="absolute bottom-1 inset-x-2 flex items-center justify-between bg-black/70 px-2 py-1 border border-cyan-500/20 rounded font-mono text-[9px] text-cyan-300">
          <span>
            {gestureData.isFist
              ? "✊ BRAKE (MAGNETIC STOP)"
              : gestureData.gestureType === "SWIPE_SPIN"
              ? "⚡ GLOBE SWIPE SPIN"
              : gestureData.handCount > 1
              ? "👐 DUAL HAND ZOOM"
              : "🖐️ ORB TILT CONTROL"}
          </span>
          <span className="text-cyan-400/80">ZOOM: {Math.round(gestureData.zoom)}%</span>
        </div>
      )}
    </div>
  );
}
