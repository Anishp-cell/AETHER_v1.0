"use client";
import { useEffect, useRef, useState, useCallback } from "react";

export interface MicroExpertSignal {
  intent: string;
  intent_confidence: number;
  emotion: string;
}

export interface TranscriptEntry {
  role: "user" | "aether";
  text: string;
  intent?: string;
  emotion?: string;
  adapters?: string[];
}

export interface AgentStatus {
  voice_engine: string;
  stt_engine: string;
  micro_expert: string;
  policy_router: string;
  vision: string;
  deep_memory: string;
  llm_engine: string;
}

export interface AetherState {
  status: "booting" | "idle" | "listening" | "thinking" | "speaking";
  mode: string;
  transcript: TranscriptEntry[];
  adapters: string[];
  agents: AgentStatus;
  micro_expert_signal: MicroExpertSignal;
  audio_energy: number;
  authRequest: { action_name: string; details: string; } | null;
  mic_muted?: boolean;
}

const DEFAULT_STATE: AetherState = {
  status: "booting",
  mode: "Push-to-Talk",
  transcript: [],
  adapters: [],
  agents: {
    voice_engine: "loading",
    stt_engine: "loading",
    micro_expert: "loading",
    policy_router: "online",
    vision: "loading",
    deep_memory: "online",
    llm_engine: "online",
  },
  micro_expert_signal: { intent: "", intent_confidence: 0, emotion: "" },
  audio_energy: 0,
  authRequest: null,
};

export function useAether() {
  const [state, setState] = useState<AetherState>(DEFAULT_STATE);
  const [energy, setEnergy] = useState(0);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => { setConnected(false); setTimeout(() => window.location.reload(), 3000); };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "state_update") {
        setState(msg.state);
      } else if (msg.type === "transcript") {
        setState((prev) => ({
          ...prev,
          transcript: [...prev.transcript, msg.entry].slice(-50),
        }));
      } else if (msg.type === "energy") {
        setEnergy(msg.value);
      } else if (msg.type === "auth_request") {
        setState((prev) => ({
          ...prev,
          authRequest: { action_name: msg.action_name, details: msg.details }
        }));
      }
    };

    return () => ws.close();
  }, []);

  const sendPTT = useCallback(
    (action: "ptt_press" | "ptt_release") => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action }));
      }
    },
    []
  );

  const sendAuthResponse = useCallback(
    (allow: boolean) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: "auth_response", allow }));
        setState((prev) => ({ ...prev, authRequest: null }));
      }
    },
    []
  );

  const toggleMute = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "toggle_mute" }));
    }
  }, []);

  return { state, energy, connected, sendPTT, sendAuthResponse, toggleMute };
}
