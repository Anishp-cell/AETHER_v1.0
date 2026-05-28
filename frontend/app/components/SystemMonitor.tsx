"use client";
import { useEffect, useRef, useState } from "react";

interface MetricHistory {
  cpu: number[];
  ram: number[];
  battery: number[];
}

interface SystemMonitorProps {
  metrics: { cpu: number; ram: number; battery: number | null; plugged: boolean | null } | null;
}

const MAX_POINTS = 30;

function MiniGraph({
  history,
  color,
  label,
  value,
  unit = "%",
}: {
  history: number[];
  color: string;
  label: string;
  value: number | null;
  unit?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    if (history.length < 2) return;

    // Fill gradient under the line
    const gradient = ctx.createLinearGradient(0, 0, 0, H);
    gradient.addColorStop(0, color.replace("1)", "0.25)"));
    gradient.addColorStop(1, color.replace("1)", "0.01)"));

    ctx.beginPath();
    history.forEach((val, i) => {
      const x = (i / (MAX_POINTS - 1)) * W;
      const y = H - (val / 100) * H;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    // Close path for fill
    ctx.lineTo(W, H);
    ctx.lineTo(0, H);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw stroke line
    ctx.beginPath();
    history.forEach((val, i) => {
      const x = (i / (MAX_POINTS - 1)) * W;
      const y = H - (val / 100) * H;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }, [history, color]);

  const pct = value !== null ? value : 0;
  const barColor =
    pct > 85 ? "rgba(239,68,68,1)" : pct > 60 ? "rgba(251,191,36,1)" : color;

  return (
    <div className="mb-3">
      <div className="flex justify-between items-center mb-1">
        <span className="text-[9px] font-mono uppercase tracking-widest text-gray-500">
          {label}
        </span>
        <span
          className="text-[10px] font-mono font-bold"
          style={{ color: barColor }}
        >
          {value !== null ? `${value.toFixed(0)}${unit}` : "—"}
        </span>
      </div>
      {/* Progress bar */}
      <div className="h-1 bg-white/5 rounded-full mb-1.5 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${barColor.replace(
              "1)",
              "0.6)"
            )}, ${barColor})`,
            boxShadow: `0 0 6px ${barColor.replace("1)", "0.4)")}`,
          }}
        />
      </div>
      {/* Mini graph */}
      <canvas
        ref={canvasRef}
        width={200}
        height={28}
        className="w-full rounded"
        style={{ opacity: 0.9 }}
      />
    </div>
  );
}

export default function SystemMonitor({ metrics }: SystemMonitorProps) {
  const [history, setHistory] = useState<MetricHistory>({
    cpu: [],
    ram: [],
    battery: [],
  });

  useEffect(() => {
    if (!metrics) return;
    setHistory((prev) => ({
      cpu: [...prev.cpu.slice(-MAX_POINTS + 1), metrics.cpu ?? 0],
      ram: [...prev.ram.slice(-MAX_POINTS + 1), metrics.ram ?? 0],
      battery: [
        ...prev.battery.slice(-MAX_POINTS + 1),
        metrics.battery ?? prev.battery[prev.battery.length - 1] ?? 0,
      ],
    }));
  }, [metrics]);

  const lastUpdate = metrics ? "Live" : "Waiting...";

  return (
    <div className="p-3">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-[9px] font-mono uppercase tracking-widest text-cyan-600">
          System Monitor
        </span>
        <div className="flex items-center gap-1">
          <div
            className="w-1.5 h-1.5 rounded-full animate-pulse"
            style={{
              background: metrics ? "rgba(52,211,153,1)" : "rgba(107,114,128,1)",
              boxShadow: metrics ? "0 0 4px rgba(52,211,153,0.6)" : "none",
            }}
          />
          <span className="text-[8px] font-mono text-gray-600">{lastUpdate}</span>
        </div>
      </div>

      <MiniGraph
        history={history.cpu}
        color="rgba(34,211,238,1)"
        label="CPU"
        value={metrics?.cpu ?? null}
      />
      <MiniGraph
        history={history.ram}
        color="rgba(168,85,247,1)"
        label="RAM"
        value={metrics?.ram ?? null}
      />
      <MiniGraph
        history={history.battery}
        color={
          metrics?.plugged
            ? "rgba(52,211,153,1)"
            : "rgba(251,191,36,1)"
        }
        label={metrics?.plugged ? "Battery ⚡" : "Battery 🔋"}
        value={metrics?.battery ?? null}
      />

      {/* Update interval notice */}
      <div className="mt-2 text-center">
        <span className="text-[8px] font-mono text-gray-700">↻ updates every 5s</span>
      </div>
    </div>
  );
}
