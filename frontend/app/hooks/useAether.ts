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

export interface CodingArtifact {
  instruction: string;
  filename: string;
  code: string;
  stdout: string;
  stderr: string;
  returncode: number;
  artifacts: Array<{
    type: "image" | "html" | "table";
    name: string;
    src?: string;
    content?: string;
    headers?: string[];
    rows?: string[][];
  }>;
  timestamp: number;
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
  latest_artifact?: CodingArtifact | null;
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
  const [cameraFrame, setCameraFrame] = useState<string | null>(null);
  const [screenCaptureFlash, setScreenCaptureFlash] = useState(false);
  const [systemMetrics, setSystemMetrics] = useState<{ cpu: number; ram: number; battery: number | null; plugged: boolean | null } | null>(null);
  const [latestArtifact, setLatestArtifact] = useState<CodingArtifact | null>(null);

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
        if (msg.state.latest_artifact) {
          setLatestArtifact(msg.state.latest_artifact);
        }
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
      } else if (msg.type === "camera_frame") {
        setCameraFrame(msg.frame);
      } else if (msg.type === "system_metrics") {
        setSystemMetrics({ cpu: msg.cpu, ram: msg.ram, battery: msg.battery, plugged: msg.plugged });
      } else if (msg.type === "screen_capture_flash") {
        setScreenCaptureFlash(true);
        setTimeout(() => setScreenCaptureFlash(false), msg.duration || 350);
      } else if (msg.type === "artifact_generated") {
        setLatestArtifact(msg.artifact);
      }
    };

    return () => ws.close();
  }, []);

  const clearArtifact = useCallback(() => {
    setLatestArtifact(null);
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

  const sendShutdown = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "shutdown" }));
    }
  }, []);

  const interruptAudio = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "interrupt" }));
    }
  }, []);

  const setMode = useCallback((mode: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "set_mode", mode }));
    }
  }, []);

  return { state, energy, connected, cameraFrame, screenCaptureFlash, systemMetrics, latestArtifact, clearArtifact, sendPTT, sendAuthResponse, toggleMute, interruptAudio, sendShutdown, setMode };
}

