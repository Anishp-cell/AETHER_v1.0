"use client";
import { useEffect, useRef } from "react";

interface Props {
  frame: string | null;
  emotionSignal?: string;
  gestureLandmarks?: Array<{ x: number; y: number; z: number }> | null;
}

export default function CameraFeed({ frame, emotionSignal, gestureLandmarks }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Draw MediaPipe Hand Skeleton connections on overlay canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (gestureLandmarks && gestureLandmarks.length === 21) {
      const w = canvas.width;
      const h = canvas.height;

      // Hand skeleton connections (21 MediaPipe points)
      const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],     // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8],     // Index
        [5, 9], [9, 10], [10, 11], [11, 12], // Middle
        [9, 13], [13, 14], [14, 15], [15, 16], // Ring
        [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
        [0, 17],
      ];

      ctx.strokeStyle = "#00F5FF";
      ctx.lineWidth = 2;
      ctx.shadowColor = "#00F5FF";
      ctx.shadowBlur = 8;

      connections.forEach(([i, j]) => {
        const p1 = gestureLandmarks[i];
        const p2 = gestureLandmarks[j];
        ctx.beginPath();
        ctx.moveTo(p1.x * w, p1.y * h);
        ctx.lineTo(p2.x * w, p2.y * h);
        ctx.stroke();
      });

      // Draw glowing joint nodes
      gestureLandmarks.forEach((pt, idx) => {
        ctx.fillStyle = idx === 4 || idx === 8 ? "#FF007F" : "#FFFFFF"; // Highlight thumb & index tip in magenta
        ctx.beginPath();
        ctx.arc(pt.x * w, pt.y * h, idx === 4 || idx === 8 ? 4 : 2.5, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }, [gestureLandmarks]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/[0.04] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${frame ? "bg-green-400 animate-pulse shadow-lg shadow-green-400/30" : "bg-red-400"}`} />
          <span
            className="text-[11px] tracking-[0.2em] uppercase"
            style={{ fontFamily: "'JetBrains Mono', monospace", color: "rgba(0,245,255,0.6)" }}
          >
            Live Vision
          </span>
        </div>
        <span className="text-[10px] font-mono text-gray-600">
          {frame ? "STREAMING" : "OFFLINE"}
        </span>
      </div>

      {/* Camera viewport */}
      <div className="flex-1 relative flex items-center justify-center overflow-hidden bg-black/40">
        {frame ? (
          <img
            src={`data:image/jpeg;base64,${frame}`}
            alt="Live Camera Feed"
            className="w-full h-full object-cover"
            style={{
              filter: "contrast(1.1) brightness(0.95)",
              imageRendering: "auto",
            }}
          />
        ) : (
          <div className="flex flex-col items-center gap-3">
            <svg className="w-10 h-10 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            <span className="text-[10px] font-mono text-gray-600 tracking-wider">
              CAMERA INITIALIZING...
            </span>
          </div>
        )}

        {/* Hand Landmark Skeleton Canvas Overlay */}
        <canvas
          ref={canvasRef}
          width={480}
          height={360}
          className="absolute inset-0 w-full h-full pointer-events-none z-10"
        />

        {/* Scanline overlay effect */}
        {frame && (
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              background: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,245,255,0.015) 2px, rgba(0,245,255,0.015) 4px)",
            }}
          />
        )}

        {/* Corner brackets overlay */}
        {frame && (
          <>
            <div className="absolute top-2 left-2 w-5 h-5 border-t border-l border-cyan-500/30" />
            <div className="absolute top-2 right-2 w-5 h-5 border-t border-r border-cyan-500/30" />
            <div className="absolute bottom-2 left-2 w-5 h-5 border-b border-l border-cyan-500/30" />
            <div className="absolute bottom-2 right-2 w-5 h-5 border-b border-r border-cyan-500/30" />
          </>
        )}

        {/* Bottom info strip */}
        {frame && (
          <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent px-3 py-2 z-20">
            <div className="flex items-center justify-between">
              <span className="text-[9px] font-mono text-cyan-400/60 tracking-wider uppercase">
                {gestureLandmarks ? "MediaPipe WASM Hands" : "MediaPipe FaceMesh"}
              </span>
              <span className="text-[9px] font-mono text-green-400/60">
                ● REC
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
