"use client";
import { useEffect, useState, useRef, useMemo } from "react";

interface Props {
  status: string;
  energy: number;
  microExpertSignal: { intent: string; intent_confidence: number; emotion: string };
  adapters: string[];
}

const COLOR_THEMES: Record<string, { primary: string; secondary: string; glow: string; text: string }> = {
  booting:   { primary: "#708090", secondary: "#475569", glow: "rgba(112,128,144,0.3)", text: "text-slate-400" },
  idle:      { primary: "#ffaa00", secondary: "#b87300", glow: "rgba(255,170,0,0.35)", text: "text-amber-500" },
  listening: { primary: "#00e6ac", secondary: "#009973", glow: "rgba(0,230,172,0.4)", text: "text-emerald-400" },
  thinking:  { primary: "#ff4d00", secondary: "#b33c00", glow: "rgba(255,77,0,0.45)", text: "text-orange-500" },
  speaking:  { primary: "#9900ff", secondary: "#6600cc", glow: "rgba(153,0,255,0.45)", text: "text-purple-400" },
  muted:     { primary: "#ef4444", secondary: "#991b1b", glow: "rgba(239,68,68,0.4)", text: "text-red-500" },
};

export default function HudOverlay({ status, energy = 0, microExpertSignal, adapters }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Mouse position tracking
  const [mousePos, setMousePos] = useState({ x: 0, y: 0, clientX: 0, clientY: 0, azimuth: 0, range: 0 });
  const [isHovering, setIsHovering] = useState(false);

  // Fluctuating values for realistic telemetry
  const [telemetryState, setTelemetryState] = useState({
    bandwidth: 4.82,
    cogIndex: 0.982,
    temp: 42.4,
    frameRate: 60,
  });

  // Animated binary data/log feed
  const [logFeed, setLogFeed] = useState<string[]>([
    "[SYS] Cognitive framework ready",
    "[MOE] Routing neural nodes online",
    "[TTS] Audio synthesis channels calibrated",
  ]);

  const activeTheme = COLOR_THEMES[status] || COLOR_THEMES.idle;

  // Handle mouse coordinate tracking and mathematical azimuth / range calculation globally
  useEffect(() => {
    const handleGlobalMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const inside = 
        e.clientX >= rect.left && 
        e.clientX <= rect.right && 
        e.clientY >= rect.top && 
        e.clientY <= rect.bottom;
      
      if (!inside) {
        setIsHovering(false);
        return;
      }
      setIsHovering(true);

      const x = e.clientX - rect.left - rect.width / 2; // X coordinate relative to center
      const y = e.clientY - rect.top - rect.height / 2; // Y coordinate relative to center
      
      const azimuth = Math.round(Math.atan2(-y, x) * (180 / Math.PI)); // Convert to standard degrees (Cartesian style)
      const normalizedAzimuth = azimuth < 0 ? azimuth + 360 : azimuth;
      const range = Math.round(Math.hypot(x, y));

      setMousePos({
        x: Math.round(x),
        y: Math.round(y),
        clientX: e.clientX - rect.left,
        clientY: e.clientY - rect.top,
        azimuth: normalizedAzimuth,
        range,
      });
    };

    window.addEventListener("mousemove", handleGlobalMouseMove);
    return () => window.removeEventListener("mousemove", handleGlobalMouseMove);
  }, []);

  // Telemetry fluctuation simulator
  useEffect(() => {
    const interval = setInterval(() => {
      setTelemetryState({
        bandwidth: Number((3.5 + Math.random() * 2.5).toFixed(2)),
        cogIndex: Number((0.975 + Math.random() * 0.02).toFixed(3)),
        temp: Number((41.0 + Math.random() * 3.5).toFixed(1)),
        frameRate: Math.round(58 + Math.random() * 4),
      });
    }, 1500);
    return () => clearInterval(interval);
  }, []);

  // System log feeder simulator based on active status
  useEffect(() => {
    const statusLogs: Record<string, string[]> = {
      listening: [
        "[VAD] Signal threshold crossed",
        "[STT] Processing acoustic frame buffers",
        "[COG] Capturing linguistic intent",
      ],
      thinking: [
        "[MOE] Querying expert adapters cascade",
        "[LLM] Dispatching DeepSeek semantic pipeline",
        "[MEM] Querying vector database structures",
      ],
      speaking: [
        "[TTS] Translating token streams to phonemes",
        "[VOX] Synthesizing RyanNeural speech wave",
        "[MEM] Recording experience index",
      ],
      idle: [
        "[SYS] Telemetry baseline stabilized",
        "[MEM] Consolidation threads optimized",
        "[SEC] Override core armed",
      ],
      muted: [
        "[SEC] Input bridge blocked",
        "[MIC] Physical audio lines isolated",
        "[SYS] Standby lock active",
      ],
    };

    const logs = statusLogs[status] || statusLogs.idle;
    const randomLog = logs[Math.floor(Math.random() * logs.length)];
    const time = new Date().toLocaleTimeString().split(" ")[0];
    
    setLogFeed((prev) => [...prev.slice(-4), `[${time}] ${randomLog}`]);
  }, [status]);

  // Audio amplitude dynamic scale factor for orb and rings
  const normalizedEnergy = Math.min(1.0, energy / 15000);
  const hudScale = 1.0 + (status === "speaking" || status === "listening" ? normalizedEnergy * 0.18 : 0);

  // Rotation directions and speeds based on active cognitive mode
  const spinClasses = useMemo(() => {
    if (status === "thinking") {
      return {
        outer: "animate-spin-fast",
        inner: "animate-spin-reverse-fast",
      };
    } else if (status === "listening" || status === "speaking") {
      return {
        outer: "animate-spin-medium",
        inner: "animate-spin-reverse-medium",
      };
    } else {
      return {
        outer: "animate-spin-slow",
        inner: "animate-spin-reverse-slow",
      };
    }
  }, [status]);

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 z-10 w-full h-full pointer-events-none select-none overflow-hidden"
    >
      {/* ── CENTRAL CONCENTRIC ROTATING HUD ORBITS ── */}
      <div 
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex items-center justify-center pointer-events-none"
        style={{ 
          transform: `translate(-50%, -50%) scale(${hudScale})`, 
          transition: "transform 0.15s cubic-bezier(0.175, 0.885, 0.32, 1.275)" 
        }}
      >
        <svg
          width="480"
          height="480"
          viewBox="0 0 500 500"
          className="w-[380px] h-[380px] md:w-[440px] md:h-[440px]"
          style={{ 
            color: activeTheme.primary, 
            filter: `drop-shadow(0 0 10px ${activeTheme.glow})` 
          }}
        >
          {/* Layer 1: Outer Ticked Compass Circle (Compass Dial) */}
          <g className={`${spinClasses.outer}`}>
            <circle
              cx="250"
              cy="250"
              r="230"
              fill="none"
              stroke="currentColor"
              strokeWidth="0.75"
              strokeDasharray="2 6"
              opacity="0.25"
            />
            <circle
              cx="250"
              cy="250"
              r="220"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeDasharray="15 120 5 30 40 40"
              opacity="0.5"
            />
            {/* Cardinal Angle Labels */}
            <text x="250" y="38" fontSize="8" fontFamily="monospace" fill="currentColor" textAnchor="middle" opacity="0.8">000° N</text>
            <text x="462" y="253" fontSize="8" fontFamily="monospace" fill="currentColor" textAnchor="start" opacity="0.8">090° E</text>
            <text x="250" y="470" fontSize="8" fontFamily="monospace" fill="currentColor" textAnchor="middle" opacity="0.8">180° S</text>
            <text x="38" y="253" fontSize="8" fontFamily="monospace" fill="currentColor" textAnchor="end" opacity="0.8">270° W</text>
          </g>

          {/* Layer 2: Middle Segmented Heavy Arc Rings */}
          <g className={`${spinClasses.inner}`}>
            <circle
              cx="250"
              cy="250"
              r="190"
              fill="none"
              stroke="currentColor"
              strokeWidth="3.5"
              strokeDasharray="60 30 10 20 80 40 10 10"
              opacity="0.35"
            />
            <circle
              cx="250"
              cy="250"
              r="178"
              fill="none"
              stroke="currentColor"
              strokeWidth="0.7"
              strokeDasharray="4 4"
              opacity="0.2"
            />
            <path
              d="M 120 250 A 130 130 0 0 1 380 250"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeDasharray="10 15"
              opacity="0.6"
            />
          </g>

          {/* Layer 3: Audio Energy Reactive Inner Arc (only visible during audio) */}
          {(status === "speaking" || status === "listening") && (
            <circle
              cx="250"
              cy="250"
              r="140"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeDasharray={`${Math.max(10, normalizedEnergy * 300)} 300`}
              strokeLinecap="round"
              opacity="0.75"
              style={{
                transform: `rotate(${energy * 0.05}deg)`,
                transformOrigin: "center",
                transition: "stroke-dasharray 0.05s ease-out"
              }}
            />
          )}

          {/* Layer 4: Deep Tech Reticles */}
          <circle
            cx="250"
            cy="250"
            r="105"
            fill="none"
            stroke="currentColor"
            strokeWidth="0.5"
            strokeDasharray="1 15"
            opacity="0.4"
          />
          <circle
            cx="250"
            cy="250"
            r="80"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.2"
            strokeDasharray="40 10 5 10"
            opacity="0.3"
          />

          {/* Dynamic HUD Target Dot Core */}
          <circle
            cx="250"
            cy="250"
            r="3"
            fill="currentColor"
            opacity={status === "thinking" ? "0.9" : "0.5"}
            className={status === "thinking" ? "animate-pulse" : ""}
          />
        </svg>
      </div>

      {/* ── HIGH-TECH DYNAMIC COORDINATE CROSSHAIR (FOLLOWS MOUSE) ── */}
      {isHovering && (
        <div
          className="absolute pointer-events-none hidden md:block"
          style={{
            left: mousePos.clientX,
            top: mousePos.clientY,
            transform: "translate(-50%, -50%)",
            color: activeTheme.primary,
            filter: `drop-shadow(0 0 4px ${activeTheme.glow})`,
          }}
        >
          {/* Target Bracket Wings */}
          <svg width="60" height="60" viewBox="0 0 60 60" className="opacity-75">
            <path d="M 5 20 L 5 5 L 20 5" fill="none" stroke="currentColor" strokeWidth="1.5" />
            <path d="M 40 5 L 55 5 L 55 20" fill="none" stroke="currentColor" strokeWidth="1.5" />
            <path d="M 5 40 L 5 55 L 20 55" fill="none" stroke="currentColor" strokeWidth="1.5" />
            <path d="M 40 55 L 55 55 L 55 40" fill="none" stroke="currentColor" strokeWidth="1.5" />
            
            {/* Center target circle */}
            <circle cx="30" cy="30" r="8" fill="none" stroke="currentColor" strokeWidth="0.75" strokeDasharray="3 3" />
            {/* Cursor tiny cross */}
            <line x1="30" y1="22" x2="30" y2="38" stroke="currentColor" strokeWidth="0.75" />
            <line x1="22" y1="30" x2="38" y2="30" stroke="currentColor" strokeWidth="0.75" />
          </svg>

          {/* Coordinate Readout Next To Cursor */}
          <div className="absolute top-8 left-8 bg-black/75 border border-white/10 rounded px-2 py-1 flex flex-col font-mono text-[8px] tracking-wider leading-none shadow-[0_0_10px_rgba(0,0,0,0.5)]">
            <span className="text-white/60 mb-0.5 uppercase">Target Coordinates</span>
            <span className="text-cyan-400 font-semibold mb-0.5">X: {mousePos.x} Y: {mousePos.y}</span>
            <span className="text-gray-400">AZ: {mousePos.azimuth}° | RG: {mousePos.range}px</span>
          </div>
        </div>
      )}

      {/* ── VERTICAL RADAR LASER SCANNER (THINKING / LISTENING MODE ONLY) ── */}
      {(status === "thinking" || status === "listening") && (
        <div 
          className="absolute left-0 w-full h-[2px] pointer-events-none animate-scanner z-0"
          style={{
            background: `linear-gradient(90deg, transparent, ${activeTheme.primary}, transparent)`,
            boxShadow: `0 0 12px ${activeTheme.primary}, 0 0 25px ${activeTheme.primary}`,
          }}
        />
      )}

      {/* ── CORNER TELEMETRY BLOCK 1: TOP-LEFT SYSTEM READOUTS ── */}
      <div 
        className={`absolute top-4 left-6 w-56 font-mono text-[10px] bg-black/30 border-l-2 border-t border-white/5 p-3 rounded-tr-xl flex flex-col gap-1 backdrop-blur-sm transition-all duration-500`}
        style={{ borderColor: `${activeTheme.primary}40`, borderLeftColor: activeTheme.primary }}
      >
        <div className="flex items-center justify-between border-b border-white/[0.04] pb-1.5 mb-1.5">
          <span className={`font-semibold tracking-[0.15em] uppercase ${activeTheme.text}`}>
            Aether Core
          </span>
          <span className="text-[8px] text-gray-500 uppercase tracking-widest">
            {status}
          </span>
        </div>
        <div className="flex justify-between items-center text-gray-400">
          <span>SECURE_CORE</span>
          <span className="text-emerald-400 font-semibold">ACTIVE</span>
        </div>
        <div className="flex justify-between items-center text-gray-400">
          <span>COG_INDEX</span>
          <span className="text-cyan-300 font-bold">{telemetryState.cogIndex}</span>
        </div>
        <div className="flex justify-between items-center text-gray-400">
          <span>DEEP_LINK</span>
          <span className="text-purple-400">STABLE</span>
        </div>
        <div className="flex justify-between items-center text-gray-400">
          <span>SYS_TEMP</span>
          <span className="text-amber-400">{telemetryState.temp}°C</span>
        </div>
        <div className="flex justify-between items-center text-gray-400">
          <span>COG_CYCLE</span>
          <span className="text-gray-500">TRM_LOOP_OK</span>
        </div>
      </div>

      {/* ── CORNER TELEMETRY BLOCK 2: TOP-RIGHT MICRO-EXPERT COGNITION ── */}
      <div 
        className="absolute top-4 right-6 w-60 font-mono text-[10px] bg-black/30 border-r-2 border-t border-white/5 p-3 rounded-tl-xl flex flex-col gap-1 backdrop-blur-sm transition-all duration-500"
        style={{ borderColor: `${activeTheme.primary}40`, borderRightColor: activeTheme.primary }}
      >
        <div className="flex items-center justify-between border-b border-white/[0.04] pb-1.5 mb-1.5">
          <span className={`font-semibold tracking-[0.15em] uppercase ${activeTheme.text}`}>
            Cognitive Signal
          </span>
          <span className="text-[8px] text-gray-500 font-bold uppercase">
            FPS: {telemetryState.frameRate}
          </span>
        </div>
        <div className="flex justify-between items-center text-gray-400">
          <span>STATE_INTENT</span>
          <span className="text-cyan-400 font-bold truncate max-w-[100px] text-right" title={microExpertSignal.intent || "idle"}>
            {microExpertSignal.intent ? microExpertSignal.intent.toUpperCase() : "STANDBY"}
          </span>
        </div>
        <div className="flex justify-between items-center text-gray-400">
          <span>INTENT_CONF</span>
          <span className="text-blue-400">
            {microExpertSignal.intent ? `${(microExpertSignal.intent_confidence * 100).toFixed(0)}%` : "0%"}
          </span>
        </div>
        <div className="flex justify-between items-center text-gray-400">
          <span>STATE_EMOTION</span>
          <span className="text-amber-400 uppercase font-semibold">
            {microExpertSignal.emotion || "NEUTRAL"}
          </span>
        </div>
        <div className="flex justify-between items-center text-gray-400">
          <span>DECISION_MODE</span>
          <span className="text-purple-400 uppercase">SYS_MOE</span>
        </div>
      </div>

      {/* ── CORNER TELEMETRY BLOCK 3: BOTTOM-LEFT LIVE LOG STREAM ── */}
      <div 
        className="absolute bottom-4 left-6 w-64 font-mono text-[9px] bg-black/35 border-l-2 border-b border-white/5 p-3 rounded-br-xl flex flex-col gap-1 backdrop-blur-sm transition-all duration-500"
        style={{ borderColor: `${activeTheme.primary}40`, borderLeftColor: activeTheme.primary }}
      >
        <div className="text-[8px] font-bold text-gray-500 uppercase tracking-widest border-b border-white/[0.04] pb-1 mb-1 flex items-center justify-between">
          <span>Neural Log Feed</span>
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
        </div>
        <div className="flex flex-col gap-0.5 leading-tight text-cyan-300/80">
          {logFeed.map((log, idx) => (
            <div key={idx} className="truncate select-none font-semibold">
              {log}
            </div>
          ))}
        </div>
      </div>

      {/* ── CORNER TELEMETRY BLOCK 4: BOTTOM-RIGHT ACTIVE COGNITIVE ADAPTERS ── */}
      <div 
        className="absolute bottom-4 right-6 w-60 font-mono text-[10px] bg-black/35 border-r-2 border-b border-white/5 p-3 rounded-bl-xl flex flex-col gap-1 backdrop-blur-sm transition-all duration-500"
        style={{ borderColor: `${activeTheme.primary}40`, borderRightColor: activeTheme.primary }}
      >
        <div className="text-[8px] font-bold text-gray-500 uppercase tracking-widest border-b border-white/[0.04] pb-1 mb-1">
          Active Cognitive Path
        </div>
        <div className="flex flex-col gap-0.5 leading-tight text-gray-400">
          <div className="flex justify-between items-center">
            <span>SIGNAL_NET</span>
            <span className="text-purple-400 text-[9px] font-semibold truncate max-w-[120px] text-right">
              {adapters.length > 0 ? adapters.join(", ") : "BASE_ROUTING"}
            </span>
          </div>
          <div className="flex justify-between items-center mt-0.5">
            <span>DATA_TX_RATE</span>
            <span className="text-cyan-400">{telemetryState.bandwidth} MB/s</span>
          </div>
          <div className="flex justify-between items-center">
            <span>PHYSICAL_LOCK</span>
            <span className={status === "muted" ? "text-red-400" : "text-green-400"}>
              {status === "muted" ? "ISOLATED" : "CONNECTED"}
            </span>
          </div>
        </div>
      </div>

      {/* Decorative technical corner frames and dots over the viewport */}
      <div className="absolute top-2 left-2 w-4 h-4 border-t border-l border-white/10 pointer-events-none" />
      <div className="absolute top-2 right-2 w-4 h-4 border-t border-r border-white/10 pointer-events-none" />
      <div className="absolute bottom-2 left-2 w-4 h-4 border-b border-l border-white/10 pointer-events-none" />
      <div className="absolute bottom-2 right-2 w-4 h-4 border-b border-r border-white/10 pointer-events-none" />
    </div>
  );
}
