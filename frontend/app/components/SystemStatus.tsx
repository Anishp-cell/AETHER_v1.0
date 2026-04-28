"use client";
import { AgentStatus } from "../hooks/useAether";

interface Props {
  agents: AgentStatus;
  adapters: string[];
  signal: { intent: string; intent_confidence: number; emotion: string };
  onShutdown: () => void;
}

const AGENT_LABELS: Record<keyof AgentStatus, { label: string; icon: string }> = {
  voice_engine: { label: "Voice Engine", icon: "🔊" },
  stt_engine: { label: "Whisper STT", icon: "🎙️" },
  micro_expert: { label: "Micro-Expert", icon: "🧬" },
  policy_router: { label: "Policy Router", icon: "🔀" },
  vision: { label: "Vision Agent", icon: "👁️" },
  deep_memory: { label: "Deep Memory", icon: "🧠" },
  llm_engine: { label: "LLM Engine", icon: "⚡" },
};

function StatusDot({ status }: { status: string }) {
  const colors: Record<string, string> = {
    online: "bg-emerald-400 shadow-emerald-400/50",
    loading: "bg-amber-400 shadow-amber-400/50 animate-pulse",
    processing: "bg-blue-400 shadow-blue-400/50 animate-pulse",
    offline: "bg-gray-600",
  };
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full shadow-lg ${colors[status] || colors.offline}`}
    />
  );
}

export default function SystemStatus({ agents, adapters, signal, onShutdown }: Props) {
  return (
    <div className="flex flex-col h-full p-4 gap-4">
      {/* Title */}
      <div className="flex items-center gap-2">
        <div className="w-1 h-5 bg-cyan-400 rounded-full" />
        <h2 className="text-xs font-mono tracking-[0.2em] uppercase text-cyan-400">
          Sub-Agents
        </h2>
      </div>

      {/* Agent List */}
      <div className="flex flex-col gap-2">
        {(Object.keys(AGENT_LABELS) as (keyof AgentStatus)[]).map((key) => (
          <div
            key={key}
            className="flex items-center gap-3 px-3 py-2 rounded-lg bg-white/[0.03] border border-white/[0.05] hover:bg-white/[0.06] transition-colors"
          >
            <span className="text-sm">{AGENT_LABELS[key].icon}</span>
            <span className="text-xs font-mono text-gray-300 flex-1">
              {AGENT_LABELS[key].label}
            </span>
            <StatusDot status={agents[key]} />
          </div>
        ))}
      </div>

      {/* Divider */}
      <div className="border-t border-white/[0.06] my-1" />

      {/* Micro-Expert Signal */}
      {signal.intent && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <div className="w-1 h-5 bg-purple-400 rounded-full" />
            <h2 className="text-xs font-mono tracking-[0.2em] uppercase text-purple-400">
              Classification
            </h2>
          </div>
          <div className="px-3 py-2 rounded-lg bg-white/[0.03] border border-white/[0.05]">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono text-gray-500 uppercase">Intent</span>
              <span className="text-xs font-mono text-cyan-300">{signal.intent}</span>
            </div>
            <div className="mt-1.5 w-full h-1 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-500"
                style={{ width: `${signal.intent_confidence * 100}%` }}
              />
            </div>
            <div className="flex items-center justify-between mt-2">
              <span className="text-[10px] font-mono text-gray-500 uppercase">Emotion</span>
              <span className="text-xs font-mono text-amber-300">{signal.emotion}</span>
            </div>
          </div>
        </div>
      )}

      {/* Divider */}
      <div className="border-t border-white/[0.06] my-1" />

      {/* Active Adapters */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <div className="w-1 h-5 bg-amber-400 rounded-full" />
          <h2 className="text-xs font-mono tracking-[0.2em] uppercase text-amber-400">
            Active Adapters
          </h2>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {adapters.length > 0 ? (
            adapters.map((a, i) => (
              <span
                key={i}
                className="px-2 py-0.5 text-[10px] font-mono rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-300"
              >
                {a}
              </span>
            ))
          ) : (
            <span className="text-[10px] font-mono text-gray-600">No adapters active</span>
          )}
        </div>
      </div>
      
      {/* Spacer */}
      <div className="flex-1" />

      {/* EMERGENCY SHUTDOWN */}
      <button
        onClick={onShutdown}
        className="w-full mt-4 py-3 bg-red-500/10 hover:bg-red-500/20 text-red-500 font-mono text-xs tracking-[0.2em] uppercase transition-all rounded-lg border border-red-500/30 hover:border-red-500 hover:shadow-[0_0_15px_rgba(239,68,68,0.3)] flex items-center justify-center gap-2"
      >
        <span className="text-sm">⏻</span> System Shutdown
      </button>
    </div>
  );
}
