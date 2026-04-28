"use client";
import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import { TranscriptEntry } from "../hooks/useAether";

interface Props {
  transcript: TranscriptEntry[];
  status: string;
}

export default function ChatPanel({ transcript, status }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript.length]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 p-4 pb-2">
        <div className="w-1 h-5 bg-cyan-400 rounded-full" />
        <h2 className="text-xs font-mono tracking-[0.2em] uppercase text-cyan-400">
          Transcript
        </h2>
        <div className="flex-1" />
        <div className="flex items-center gap-1.5">
          <span
            className={`inline-block w-1.5 h-1.5 rounded-full ${
              status === "listening"
                ? "bg-green-400 animate-pulse"
                : status === "speaking"
                ? "bg-purple-400 animate-pulse"
                : status === "thinking"
                ? "bg-orange-400 animate-pulse"
                : "bg-gray-600"
            }`}
          />
          <span className="text-[10px] font-mono text-gray-500 uppercase">{status}</span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-2 space-y-3 scrollbar-thin">
        {transcript.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <p className="text-xs font-mono text-gray-600 animate-pulse">
              Awaiting conversation...
            </p>
          </div>
        )}

        {transcript.map((entry, i) => (
          <div
            key={i}
            className={`flex flex-col gap-1 animate-fadeIn ${
              entry.role === "user" ? "items-end" : "items-start"
            }`}
          >
            {/* Classification badges (user messages only) */}
            {entry.role === "user" && (entry.intent || entry.emotion) && (
              <div className="flex gap-1.5 mb-0.5">
                {entry.intent && (
                  <span className="px-1.5 py-0.5 text-[9px] font-mono rounded bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
                    {entry.intent}
                  </span>
                )}
                {entry.emotion && entry.emotion !== "neutral" && (
                  <span className="px-1.5 py-0.5 text-[9px] font-mono rounded bg-amber-500/10 border border-amber-500/20 text-amber-400">
                    {entry.emotion}
                  </span>
                )}
              </div>
            )}

            {/* Adapter badges (aether messages only) */}
            {entry.role === "aether" && entry.adapters && entry.adapters.length > 0 && (
              <div className="flex gap-1 flex-wrap mb-0.5">
                {entry.adapters.map((a, j) => (
                  <span
                    key={j}
                    className="px-1.5 py-0.5 text-[9px] font-mono rounded bg-purple-500/10 border border-purple-500/20 text-purple-400"
                  >
                    {a}
                  </span>
                ))}
              </div>
            )}

            {/* Message bubble */}
            <div
              className={`max-w-[85%] px-3.5 py-2.5 rounded-2xl text-sm leading-relaxed prose prose-invert max-w-none ${
                entry.role === "user"
                  ? "bg-cyan-500/10 border border-cyan-500/15 text-cyan-100 rounded-br-md"
                  : "bg-white/[0.04] border border-white/[0.06] text-gray-200 rounded-bl-md"
              }`}
            >
              {entry.role === "aether" && (
                <span className="text-[10px] font-mono text-purple-400 block mb-1">AETHER</span>
              )}
              {entry.role === "user" ? (
                entry.text
              ) : (
                <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
                  {entry.text}
                </ReactMarkdown>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
