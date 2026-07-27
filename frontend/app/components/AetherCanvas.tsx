"use client";
import React, { useState } from "react";
import { CodingArtifact } from "../hooks/useAether";

interface AetherCanvasProps {
  artifact: CodingArtifact | null;
  onClose: () => void;
}

export default function AetherCanvas({ artifact, onClose }: AetherCanvasProps) {
  const [activeTab, setActiveTab] = useState<"render" | "code" | "terminal">("render");
  const [isMaximized, setIsMaximized] = useState(false);
  const [copied, setCopied] = useState(false);

  if (!artifact) return null;

  const hasVisualArtifacts = artifact.artifacts && artifact.artifacts.length > 0;

  const handleCopyCode = () => {
    navigator.clipboard.writeText(artifact.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadCode = () => {
    const blob = new Blob([artifact.code], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = artifact.filename || "aether_script.py";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div
      className={`fixed z-50 transition-all duration-500 ease-out flex flex-col glass-panel shadow-[0_0_50px_rgba(0,245,255,0.2)] border border-cyan-500/30 overflow-hidden backdrop-blur-xl ${
        isMaximized
          ? "inset-4 rounded-2xl"
          : "bottom-6 right-6 w-[720px] h-[540px] rounded-xl"
      }`}
      style={{
        background: "rgba(10, 15, 30, 0.92)",
      }}
    >
      {/* Canvas Top Bar */}
      <header className="h-12 border-b border-white/[0.08] px-4 flex items-center justify-between bg-black/40 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse shadow-[0_0_8px_#00f5ff]" />
          <span className="text-xs font-mono tracking-widest text-cyan-300 uppercase font-semibold">
            AETHERCODER CANVAS
          </span>
          <span className="text-[10px] font-mono text-gray-400 bg-white/[0.05] px-2 py-0.5 rounded border border-white/[0.06]">
            {artifact.filename}
          </span>
          {artifact.returncode === 0 ? (
            <span className="text-[9px] font-mono bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded border border-emerald-500/30">
              EXIT 0
            </span>
          ) : (
            <span className="text-[9px] font-mono bg-red-500/20 text-red-300 px-1.5 py-0.5 rounded border border-red-500/30">
              EXIT {artifact.returncode}
            </span>
          )}
        </div>

        {/* Canvas Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsMaximized(!isMaximized)}
            className="p-1.5 text-gray-400 hover:text-cyan-300 transition-colors text-xs font-mono"
            title={isMaximized ? "Restore Size" : "Maximize"}
          >
            {isMaximized ? "🗗" : "🗖"}
          </button>
          <button
            onClick={onClose}
            className="p-1.5 text-gray-400 hover:text-red-400 transition-colors text-xs font-mono font-bold"
            title="Close Canvas"
          >
            ✕
          </button>
        </div>
      </header>

      {/* Task Instruction Banner */}
      <div className="px-4 py-2 bg-cyan-950/30 border-b border-cyan-500/10 flex items-center justify-between text-xs font-mono text-cyan-200/80">
        <div className="truncate pr-4">
          <span className="text-cyan-400 font-semibold mr-2">TASK:</span>
          "{artifact.instruction}"
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <button
            onClick={handleCopyCode}
            className="px-2.5 py-1 rounded bg-white/[0.05] hover:bg-cyan-500/20 border border-white/[0.08] text-[10px] text-cyan-300 transition-all"
          >
            {copied ? "COPIED ✓" : "COPY CODE"}
          </button>
          <button
            onClick={handleDownloadCode}
            className="px-2.5 py-1 rounded bg-white/[0.05] hover:bg-cyan-500/20 border border-white/[0.08] text-[10px] text-cyan-300 transition-all"
          >
            DOWNLOAD .PY
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-white/[0.06] bg-black/20 px-4 pt-2 gap-2 flex-shrink-0">
        <button
          onClick={() => setActiveTab("render")}
          className={`px-4 py-1.5 rounded-t-lg text-xs font-mono transition-all border-t border-x ${
            activeTab === "render"
              ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/40 font-semibold shadow-[0_-5px_15px_rgba(0,245,255,0.1)]"
              : "text-gray-400 hover:text-gray-200 border-transparent"
          }`}
        >
          📊 VISUAL OUTPUT ({artifact.artifacts?.length || 0})
        </button>
        <button
          onClick={() => setActiveTab("code")}
          className={`px-4 py-1.5 rounded-t-lg text-xs font-mono transition-all border-t border-x ${
            activeTab === "code"
              ? "bg-purple-500/20 text-purple-300 border-purple-500/40 font-semibold shadow-[0_-5px_15px_rgba(168,85,247,0.1)]"
              : "text-gray-400 hover:text-gray-200 border-transparent"
          }`}
        >
          💻 PYTHON SOURCE
        </button>
        <button
          onClick={() => setActiveTab("terminal")}
          className={`px-4 py-1.5 rounded-t-lg text-xs font-mono transition-all border-t border-x ${
            activeTab === "terminal"
              ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40 font-semibold shadow-[0_-5px_15px_rgba(16,185,129,0.1)]"
              : "text-gray-400 hover:text-gray-200 border-transparent"
          }`}
        >
          📟 TERMINAL LOGS
        </button>
      </div>

      {/* Main Tab Content Viewport */}
      <div className="flex-1 overflow-auto p-4 relative bg-black/40">
        {/* 1. VISUAL RENDER TAB */}
        {activeTab === "render" && (
          <div className="h-full flex flex-col gap-4">
            {!hasVisualArtifacts ? (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-6 border border-dashed border-white/10 rounded-xl bg-white/[0.01]">
                <div className="text-3xl mb-2">⚡</div>
                <div className="text-xs font-mono text-gray-300 mb-1">Execution Completed Without Image/HTML Files</div>
                <div className="text-[11px] font-mono text-gray-500 max-w-md">
                  The script executed successfully and generated terminal text output. Switch to the <span className="text-cyan-400 font-semibold">TERMINAL LOGS</span> tab to view full stdout/stderr.
                </div>
              </div>
            ) : (
              artifact.artifacts.map((art, idx) => (
                <div key={idx} className="bg-black/60 rounded-xl border border-white/10 overflow-hidden p-3 flex flex-col gap-2">
                  <div className="flex items-center justify-between border-b border-white/5 pb-2">
                    <span className="text-xs font-mono text-cyan-300 font-semibold uppercase">
                      {art.name} ({art.type})
                    </span>
                    {art.type === "image" && art.src && (
                      <a
                        href={art.src}
                        download={art.name}
                        className="text-[10px] font-mono text-cyan-400 hover:underline"
                      >
                        SAVE IMAGE ↓
                      </a>
                    )}
                  </div>

                  {/* Render Image (Matplotlib chart, graph) */}
                  {art.type === "image" && art.src && (
                    <div className="flex justify-center items-center bg-black/80 rounded-lg p-2 overflow-auto max-h-[380px]">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={art.src}
                        alt={art.name}
                        className="max-w-full h-auto object-contain rounded border border-white/10 shadow-xl"
                      />
                    </div>
                  )}

                  {/* Render HTML Preview */}
                  {art.type === "html" && art.content && (
                    <div className="h-96 rounded-lg overflow-hidden border border-white/10 bg-white">
                      <iframe
                        srcDoc={art.content}
                        title={art.name}
                        className="w-full h-full border-none"
                      />
                    </div>
                  )}

                  {/* Render CSV Interactive Table */}
                  {art.type === "table" && art.headers && art.rows && (
                    <div className="overflow-x-auto max-h-72 rounded-lg border border-white/10 bg-black/80">
                      <table className="w-full text-left border-collapse font-mono text-xs">
                        <thead>
                          <tr className="bg-cyan-950/40 border-b border-cyan-500/20 text-cyan-300">
                            {art.headers.map((h, hIdx) => (
                              <th key={hIdx} className="px-3 py-2 text-[11px] font-semibold tracking-wider border-r border-white/5 last:border-r-0">
                                {h}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {art.rows.map((row, rIdx) => (
                            <tr key={rIdx} className="border-b border-white/5 hover:bg-white/[0.03] transition-colors">
                              {row.map((cell, cIdx) => (
                                <td key={cIdx} className="px-3 py-1.5 text-[11px] text-gray-300 border-r border-white/5 last:border-r-0">
                                  {cell}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        )}

        {/* 2. PYTHON SOURCE CODE TAB */}
        {activeTab === "code" && (
          <div className="h-full rounded-xl border border-purple-500/20 bg-black/80 p-4 font-mono text-xs overflow-auto">
            <pre className="text-purple-200 leading-relaxed">
              {artifact.code.split("\n").map((line, lIdx) => (
                <div key={lIdx} className="flex gap-4 hover:bg-purple-500/10 px-1 rounded">
                  <span className="text-gray-600 select-none text-right w-8">{lIdx + 1}</span>
                  <span className="flex-1">{line}</span>
                </div>
              ))}
            </pre>
          </div>
        )}

        {/* 3. TERMINAL LOGS TAB */}
        {activeTab === "terminal" && (
          <div className="h-full rounded-xl border border-emerald-500/20 bg-black/90 p-4 font-mono text-xs overflow-auto flex flex-col gap-3">
            <div>
              <div className="text-emerald-400 font-semibold mb-1 uppercase tracking-wider text-[11px]">
                Standard Output (STDOUT):
              </div>
              <pre className="text-gray-300 bg-white/[0.02] p-3 rounded-lg border border-white/5 whitespace-pre-wrap">
                {artifact.stdout || "[No stdout output printed]"}
              </pre>
            </div>

            {artifact.stderr && (
              <div>
                <div className="text-red-400 font-semibold mb-1 uppercase tracking-wider text-[11px]">
                  Standard Error (STDERR):
                </div>
                <pre className="text-red-300 bg-red-950/30 p-3 rounded-lg border border-red-500/20 whitespace-pre-wrap">
                  {artifact.stderr}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
