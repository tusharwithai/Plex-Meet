"use client";

import React, { useEffect, useRef } from "react";
import { useMeetStore } from "@/store/useMeetStore";

interface Props {
  isConnected: boolean;
}

export default function SignLanguageOverlay({ isConnected }: Props) {
  const {
    signLetter,
    signConfidence,
    signWordBuffer,
    signTranscript,
    clearSignTranscript,
  } = useMeetStore();

  const transcriptEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll transcript to bottom
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [signTranscript]);

  const confidencePct = Math.round(signConfidence * 100);
  const confColor =
    confidencePct >= 80 ? "#34d399" :
    confidencePct >= 60 ? "#fbbf24" : "#f87171";

  return (
    <div
      style={{
        width: "260px",
        background: "rgba(28, 30, 36, 0.90)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        border: "1px solid rgba(255,255,255,0.10)",
        borderRadius: "18px",
        padding: "16px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px rgba(138,180,248,0.08)",
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        fontFamily: "'Inter', system-ui, sans-serif",
        userSelect: "none",
      }}
    >
      {/* ── Header ─────────────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "14px" }}>🤟</span>
          <span style={{ color: "#e8eaed", fontSize: "13px", fontWeight: 600, letterSpacing: "0.02em" }}>
            Sign Language
          </span>
        </div>

        {/* Connection status */}
        <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
          <div
            style={{
              width: "7px",
              height: "7px",
              borderRadius: "50%",
              background: isConnected ? "#34d399" : "#f87171",
              boxShadow: isConnected ? "0 0 6px #34d39988" : "0 0 6px #f8717188",
              transition: "all 0.3s ease",
            }}
          />
          <span style={{ color: isConnected ? "#34d399" : "#f87171", fontSize: "11px", fontWeight: 500 }}>
            {isConnected ? "Live" : "Offline"}
          </span>
        </div>
      </div>

      {/* ── Letter Display ─────────────────────────────────────────── */}
      <div
        style={{
          background: "rgba(138,180,248,0.06)",
          border: `1px solid ${signLetter ? "rgba(138,180,248,0.25)" : "rgba(255,255,255,0.06)"}`,
          borderRadius: "14px",
          padding: "12px 16px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          transition: "border-color 0.2s ease",
          minHeight: "70px",
        }}
      >
        {/* Big letter */}
        <div
          style={{
            fontSize: "52px",
            fontWeight: 700,
            color: signLetter ? "#8ab4f8" : "rgba(255,255,255,0.15)",
            lineHeight: 1,
            letterSpacing: "-2px",
            transition: "all 0.15s ease",
            textShadow: signLetter ? "0 0 20px rgba(138,180,248,0.4)" : "none",
            minWidth: "56px",
            textAlign: "center",
          }}
        >
          {signLetter || "—"}
        </div>

        {/* Confidence */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "6px" }}>
          <span style={{ color: confColor, fontSize: "14px", fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
            {signLetter ? `${confidencePct}%` : ""}
          </span>
          {signLetter && (
            <div
              style={{
                width: "60px",
                height: "4px",
                background: "rgba(255,255,255,0.1)",
                borderRadius: "2px",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${confidencePct}%`,
                  background: confColor,
                  borderRadius: "2px",
                  transition: "width 0.2s ease, background 0.2s ease",
                }}
              />
            </div>
          )}
        </div>
      </div>

      {/* ── Word Buffer ─────────────────────────────────────────────── */}
      <div
        style={{
          background: "rgba(255,255,255,0.04)",
          borderRadius: "10px",
          padding: "8px 12px",
          minHeight: "34px",
          display: "flex",
          alignItems: "center",
          gap: "4px",
        }}
      >
        <span style={{ color: "rgba(255,255,255,0.35)", fontSize: "11px", flexShrink: 0 }}>Building:</span>
        <span
          style={{
            color: "#fbbf24",
            fontSize: "14px",
            fontWeight: 600,
            letterSpacing: "3px",
            fontFamily: "monospace",
          }}
        >
          {signWordBuffer || <span style={{ color: "rgba(255,255,255,0.2)", letterSpacing: "normal", fontFamily: "inherit" }}>—</span>}
        </span>
        {signWordBuffer && (
          <span
            style={{
              display: "inline-block",
              width: "2px",
              height: "15px",
              background: "#fbbf24",
              marginLeft: "2px",
              borderRadius: "1px",
              animation: "blink 1s step-end infinite",
            }}
          />
        )}
      </div>

      {/* ── Transcript ──────────────────────────────────────────────── */}
      <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span style={{ color: "rgba(255,255,255,0.4)", fontSize: "11px", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Transcript
          </span>
          {signTranscript.length > 0 && (
            <button
              onClick={clearSignTranscript}
              style={{
                background: "none",
                border: "none",
                color: "rgba(255,255,255,0.3)",
                fontSize: "11px",
                cursor: "pointer",
                padding: "2px 6px",
                borderRadius: "4px",
                transition: "color 0.2s",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "#f87171")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "rgba(255,255,255,0.3)")}
            >
              clear
            </button>
          )}
        </div>

        <div
          style={{
            background: "rgba(255,255,255,0.04)",
            borderRadius: "10px",
            padding: "8px 12px",
            minHeight: "64px",
            maxHeight: "100px",
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            gap: "2px",
          }}
        >
          {signTranscript.length === 0 ? (
            <span style={{ color: "rgba(255,255,255,0.2)", fontSize: "12px", fontStyle: "italic" }}>
              Speak a sign to begin…
            </span>
          ) : (
            <p style={{ margin: 0, color: "#e8eaed", fontSize: "13px", lineHeight: 1.6, wordBreak: "break-word" }}>
              {signTranscript.join(" ")}
            </p>
          )}
          <div ref={transcriptEndRef} />
        </div>
      </div>

      {/* ── Hint ────────────────────────────────────────────────────── */}
      {!isConnected && (
        <p style={{ margin: 0, color: "#f87171", fontSize: "11px", textAlign: "center" }}>
          Start <code style={{ background: "rgba(255,255,255,0.1)", padding: "1px 4px", borderRadius: "3px" }}>sign_server.py</code> to connect
        </p>
      )}

      {/* Cursor blink keyframe */}
      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }`}</style>
    </div>
  );
}
