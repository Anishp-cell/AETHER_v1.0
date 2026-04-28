"use client";
import { useEffect, useRef, useCallback } from "react";

interface Props {
  energy: number;
  status: string;
  isMuted: boolean;
  onToggleMute: () => void;
  onInterrupt: () => void;
}

export default function Waveform({ energy, status, isMuted, onToggleMute, onInterrupt }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const barsRef = useRef<number[]>(new Array(64).fill(0));
  const animRef = useRef<number>(0);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const bars = barsRef.current;
    const barCount = bars.length;
    const barWidth = w / barCount;
    const gap = 2;

    // Normalize energy to 0–1 range (max ~32768 from int16 audio)
    const normalizedEnergy = Math.min(energy / 15000, 1);

    // Update bars with smooth decay
    for (let i = 0; i < barCount; i++) {
      const target =
        status === "listening" || status === "speaking"
          ? normalizedEnergy * (0.4 + Math.random() * 0.6) * h * 0.8
          : Math.random() * h * 0.05; // idle shimmer
      bars[i] += (target - bars[i]) * 0.15;
    }

    // Color based on status
    let gradient: CanvasGradient;
    if (status === "listening") {
      gradient = ctx.createLinearGradient(0, h, 0, 0);
      gradient.addColorStop(0, "rgba(0, 255, 136, 0.1)");
      gradient.addColorStop(1, "rgba(0, 255, 136, 0.8)");
    } else if (status === "speaking") {
      gradient = ctx.createLinearGradient(0, h, 0, 0);
      gradient.addColorStop(0, "rgba(170, 68, 255, 0.1)");
      gradient.addColorStop(1, "rgba(170, 68, 255, 0.8)");
    } else if (status === "thinking") {
      gradient = ctx.createLinearGradient(0, h, 0, 0);
      gradient.addColorStop(0, "rgba(255, 136, 0, 0.1)");
      gradient.addColorStop(1, "rgba(255, 136, 0, 0.8)");
    } else {
      gradient = ctx.createLinearGradient(0, h, 0, 0);
      gradient.addColorStop(0, "rgba(0, 245, 255, 0.05)");
      gradient.addColorStop(1, "rgba(0, 245, 255, 0.3)");
    }

    ctx.fillStyle = gradient;

    // Draw bars mirrored from center
    const centerY = h / 2;
    for (let i = 0; i < barCount; i++) {
      const barH = bars[i] / 2;
      const x = i * barWidth + gap / 2;
      const bw = barWidth - gap;
      // Top half
      ctx.fillRect(x, centerY - barH, bw, barH);
      // Bottom half (mirror)
      ctx.fillRect(x, centerY, bw, barH);
    }

    animRef.current = requestAnimationFrame(draw);
  }, [energy, status]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.width = canvas.offsetWidth * 2;
      canvas.height = canvas.offsetHeight * 2;
    }
    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [draw]);

  return (
    <div className="flex items-center gap-4 h-full px-4">
      {/* Mute Toggle Button */}
      <button
        onClick={onToggleMute}
        title={isMuted ? "Unmute Microphone" : "Mute Microphone"}
        className={`flex-shrink-0 w-14 h-14 rounded-2xl border flex items-center justify-center transition-all duration-300 cursor-pointer select-none ${
          isMuted
            ? "bg-red-500/20 border-red-500/50 shadow-[0_0_15px_rgba(239,68,68,0.3)] shadow-red-500/20"
            : "bg-white/[0.03] border-white/10 hover:bg-white/[0.06] hover:border-cyan-500/30"
        }`}
      >
        {isMuted ? (
          <svg className="w-6 h-6 text-red-400" fill="currentColor" viewBox="0 0 24 24">
            <path d="M19 11h-1.7c0 .74-.16 1.43-.43 2.05l1.23 1.23c.56-.98.9-2.09.9-3.28zm-4.02 3.28c-.92.83-2.16 1.35-3.48 1.49v2.12c1.82-.14 3.42-1.01 4.54-2.25l-1.06-1.36zM12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm-1.02-3.28-1.23-1.23A4.956 4.956 0 0 0 9.3 11H7.6c0 1.19.34 2.3.9 3.28l1.06-1.36-1.06 1.36A6.973 6.973 0 0 0 12 17.92v3.08h2v-3.08c.55-.05 1.08-.18 1.58-.37l-1.23-1.23c-.41.13-.85.22-1.35.26v-2.12c-.67-.07-1.29-.29-1.84-.61l1.23-1.23c.33.2.7.35 1.11.41l-1.54-1.54zm-8.73-8.45L1 3.54l16.46 16.46 1.27-1.27L2.25 2.27z" />
          </svg>
        ) : (
          <svg className="w-6 h-6 text-gray-400" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
            <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
          </svg>
        )}
      </button>

      {/* Stop / Interrupt Button */}
      <button
        onClick={onInterrupt}
        title="Stop Aether"
        className="flex-shrink-0 w-14 h-14 rounded-2xl border bg-white/[0.03] border-white/10 hover:bg-white/[0.06] hover:border-red-500/30 flex items-center justify-center transition-all duration-300 cursor-pointer select-none"
      >
        <svg className="w-6 h-6 text-red-400" fill="currentColor" viewBox="0 0 24 24">
          <path d="M6 6h12v12H6z" />
        </svg>
      </button>

      {/* Waveform canvas */}
      <div className="flex-1 h-full relative">
        <canvas
          ref={canvasRef}
          className={`w-full h-full transition-opacity duration-500 ${isMuted ? "opacity-20" : "opacity-100"}`}
          style={{ imageRendering: "auto" }}
        />
        {/* Center line */}
        <div className="absolute inset-x-0 top-1/2 h-px bg-white/[0.05]" />
      </div>

      {/* Status indicator */}
      <div className="flex-shrink-0 flex flex-col items-center gap-1 w-12">
        <span className="text-[10px] font-mono text-gray-500 uppercase tracking-wider">
          MIC
        </span>
        <span
          className={`text-[10px] font-mono ${
            isMuted ? "text-red-400" : "text-green-400"
          }`}
        >
          {isMuted ? "MUTED" : "LIVE"}
        </span>
      </div>
    </div>
  );
}
