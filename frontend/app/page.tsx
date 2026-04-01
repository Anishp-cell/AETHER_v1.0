"use client";
import dynamic from "next/dynamic";
import { useAether } from "./hooks/useAether";
import SystemStatus from "./components/SystemStatus";
import ChatPanel from "./components/ChatPanel";
import Waveform from "./components/Waveform";
import { useEffect, useCallback } from "react";

// Dynamic import for Three.js (no SSR)
const HologramOrb = dynamic(() => import("./components/HologramOrb"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full">
      <div className="text-xs font-mono text-gray-600 animate-pulse">Loading 3D Engine...</div>
    </div>
  ),
});

export default function Home() {
  const { state, energy, connected, sendAuthResponse, toggleMute } = useAether();

  // Global keyboard Mute Toggle (Spacebar)
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        toggleMute();
      }
    },
    [toggleMute]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [handleKeyDown]);

  return (
    <main className="h-screen w-screen bg-grid relative overflow-hidden">
      {/* Ambient background glow */}
      <div className="absolute inset-0 ambient-gradient pointer-events-none" />

      {/* Top bar */}
      <header className="absolute top-0 inset-x-0 h-12 z-20 flex items-center justify-between px-6 border-b border-white/[0.04]">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-lg shadow-cyan-400/30 animate-pulse" />
          <span
            className="text-xs tracking-[0.3em] uppercase"
            style={{ fontFamily: "'JetBrains Mono', monospace", color: "rgba(0,245,255,0.7)" }}
          >
            AETHER v3.0
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-[10px] font-mono text-gray-600 uppercase tracking-wider">
            System-Level MoE Runtime
          </span>
          <div className="flex items-center gap-1.5">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                connected ? "bg-emerald-400 shadow-emerald-400/30" : "bg-red-400"
              }`}
            />
            <span className="text-[10px] font-mono text-gray-500">
              {connected ? "CONNECTED" : "OFFLINE"}
            </span>
          </div>
        </div>
      </header>

      {/* Main 3-column layout */}
      <div className="absolute inset-0 top-12 bottom-20 flex">
        {/* LEFT: System Status sidebar */}
        <aside className="w-64 flex-shrink-0 glass-panel border-r border-white/[0.04] overflow-y-auto">
          <SystemStatus
            agents={state.agents}
            adapters={state.adapters}
            signal={state.micro_expert_signal}
          />
        </aside>

        {/* CENTER: Hologram Orb */}
        <section className="flex-1 relative">
          <HologramOrb status={state.status} isMuted={Boolean(state.mic_muted)} />
        </section>

        {/* RIGHT: Chat Transcript */}
        <aside className="w-96 flex-shrink-0 glass-panel glow-purple border-l border-white/[0.04] overflow-hidden">
          <ChatPanel transcript={state.transcript} status={state.status} />
        </aside>
      </div>

      {/* BOTTOM: Waveform + Mute */}
      <footer className="absolute bottom-0 inset-x-0 h-20 glass-panel border-t border-white/[0.04]">
        <Waveform
          energy={energy}
          status={state.status}
          isMuted={Boolean(state.mic_muted)}
          onToggleMute={toggleMute}
        />
      </footer>

      {/* SECURITY AUTHORIZATION MODAL */}
      {state.authRequest && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="w-[500px] bg-black/80 border border-red-500/50 shadow-[0_0_50px_rgba(239,68,68,0.2)] rounded-2xl overflow-hidden flex flex-col">
            <div className="bg-red-500/10 border-b border-red-500/30 p-4 flex items-center justify-center">
              <span className="text-red-400 font-mono tracking-widest text-sm uppercase animate-pulse">
                ⚠️ System Override Requested ⚠️
              </span>
            </div>
            
            <div className="p-6 flex flex-col gap-4">
              <div>
                <span className="text-gray-500 font-mono text-xs tracking-wider uppercase">Action</span>
                <h3 className="text-white font-mono text-lg mt-1">{state.authRequest.action_name}</h3>
              </div>
              
              <div>
                <span className="text-gray-500 font-mono text-xs tracking-wider uppercase">Target / Details</span>
                <div className="mt-1 p-3 bg-white/5 border border-white/10 rounded-lg text-gray-300 font-mono text-sm break-words whitespace-pre-wrap max-h-48 overflow-y-auto">
                  {state.authRequest.details}
                </div>
              </div>

              <div className="flex items-center gap-4 mt-4">
                <button
                  onClick={() => sendAuthResponse(false)}
                  className="flex-1 py-3 bg-white/5 hover:bg-white/10 text-white font-mono text-xs tracking-wider transition-colors rounded-lg border border-white/10"
                >
                  [ DENY ]
                </button>
                <button
                  onClick={() => sendAuthResponse(true)}
                  className="flex-1 py-3 bg-red-500/20 hover:bg-red-500/30 text-red-400 font-mono text-xs tracking-wider transition-colors rounded-lg border border-red-500/50 shadow-[0_0_15px_rgba(239,68,68,0.3)]"
                >
                  [ AUTHORIZE ]
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
