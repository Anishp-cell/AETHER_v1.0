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

    const landmarks = gestureData.landmarks;

    if (landmarks && landmarks.length === 21) {
      const w = canvas.width;
      const h = canvas.height;

      // 21 MediaPipe Hand Connections
      const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],     // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8],     // Index
        [5, 9], [9, 10], [10, 11], [11, 12], // Middle
        [9, 13], [13, 14], [14, 15], [15, 16], // Ring
        [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
        [0, 17],
      ];

      // Draw neon cyan skeleton connections
      ctx.strokeStyle = "#00F5FF";
      ctx.lineWidth = 2.5;
      ctx.shadowColor = "#00F5FF";
      ctx.shadowBlur = 12;

      connections.forEach(([i, j]) => {
        const p1 = landmarks[i];
        const p2 = landmarks[j];
        ctx.beginPath();
        ctx.moveTo(p1.x * w, p1.y * h);
        ctx.lineTo(p2.x * w, p2.y * h);
        ctx.stroke();
      });

      // Draw glowing joint nodes (Highlight thumb tip & index tip in neon magenta)
      landmarks.forEach((pt, idx) => {
        const isTip = idx === 4 || idx === 8;
        ctx.fillStyle = isTip ? "#FF007F" : "#FFFFFF";
        ctx.shadowColor = isTip ? "#FF007F" : "#00F5FF";
        ctx.shadowBlur = isTip ? 16 : 8;

        ctx.beginPath();
        ctx.arc(pt.x * w, pt.y * h, isTip ? 5 : 3, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw pinch distance indicator line between thumb tip (4) and index tip (8)
      const thumb = landmarks[4];
      const index = landmarks[8];
      ctx.strokeStyle = "rgba(255, 0, 127, 0.8)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(thumb.x * w, thumb.y * h);
      ctx.lineTo(index.x * w, index.y * h);
      ctx.stroke();
      ctx.setLineDash([]);
    } else {
      // Searching state text
      ctx.fillStyle = "rgba(0, 245, 255, 0.4)";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText("WAVE HAND IN FRONT OF CAMERA", canvas.width / 2, canvas.height / 2);
    }
  }, [gestureData]);

  return (
    <div className="relative w-full h-full border border-cyan-500/30 rounded-lg overflow-hidden bg-black/80 flex flex-col">
      {/* Header Bar */}
      <div className="px-3 py-1.5 bg-black/60 border-b border-cyan-500/20 flex items-center justify-between z-10">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${gestureData.isTracking ? "bg-cyan-400 animate-ping" : "bg-yellow-400"}`} />
          <span className="text-[10px] font-mono text-cyan-300 tracking-wider uppercase">
            Digital Hand Interface
          </span>
        </div>
        <span className="text-[9px] font-mono text-cyan-400/70">
          {gestureData.isTracking ? "TRACKING ACTIVE" : "SEARCHING"}
        </span>
      </div>

      {/* Canvas */}
      <canvas ref={canvasRef} width={240} height={180} className="w-full h-full object-cover" />

      {/* Footer Telemetry Overlay */}
      {gestureData.isTracking && (
        <div className="absolute bottom-1 inset-x-2 flex items-center justify-between bg-black/60 px-2 py-1 border border-cyan-500/20 rounded font-mono text-[9px] text-cyan-400">
          <span>PITCH: {Math.round(gestureData.rotationX * (180 / Math.PI))}°</span>
          <span>YAW: {Math.round(gestureData.rotationY * (180 / Math.PI))}°</span>
          <span>ZOOM: {Math.round(gestureData.zoom)}%</span>
        </div>
      )}
    </div>
  );
}
