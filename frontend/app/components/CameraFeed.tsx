"use client";

interface Props {
  frame: string | null;
  emotionSignal?: string;
}

export default function CameraFeed({ frame, emotionSignal }: Props) {
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
          <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent px-3 py-2">
            <div className="flex items-center justify-between">
              <span className="text-[9px] font-mono text-cyan-400/60 tracking-wider uppercase">
                MediaPipe FaceMesh
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
