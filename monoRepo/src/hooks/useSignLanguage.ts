"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import { LocalParticipant, Track } from "livekit-client";
import { useMeetStore } from "@/store/useMeetStore";

// ── Configuration ───────────────────────────────────────────────────────────────
const WS_URL          = "ws://localhost:8765";
const FRAME_INTERVAL  = 150;   // ms between frames sent to Python
const LETTER_HOLD_MS  = 500;   // ms a letter must be stable to be accepted
const WORD_TIMEOUT_MS = 1500;  // ms of no detection before speaking the word

// ── Types ───────────────────────────────────────────────────────────────────────
interface WsMessage {
  letter:     string | null;
  confidence: number;
}

export function useSignLanguage(localParticipant: LocalParticipant | undefined) {
  const {
    isSignLanguageEnabled,
    setSignLetter,
    setSignConfidence,
    setSignWordBuffer,
    appendSignTranscript,
  } = useMeetStore();

  const [isConnected, setIsConnected] = useState(false);

  // ── Refs (survive re-renders without triggering them) ───────────────────────
  const wsRef          = useRef<WebSocket | null>(null);
  const hiddenVideoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef      = useRef<HTMLCanvasElement | null>(null);
  const frameTimerRef  = useRef<number | null>(null);
  const wordTimerRef   = useRef<ReturnType<typeof setTimeout> | null>(null);
  const holdTimerRef   = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Letter accumulation
  const wordBufferRef    = useRef<string>("");   // building word (e.g. "HEL")
  const pendingLetterRef = useRef<string | null>(null);  // letter waiting for hold
  const lastAddedRef     = useRef<string>("");   // last letter added to buffer

  // ── Speak a completed word via TTS ─────────────────────────────────────────
  const speakWord = useCallback((word: string) => {
    if (!word.trim()) return;
    appendSignTranscript(word);
    window.speechSynthesis.cancel();
    const utt  = new SpeechSynthesisUtterance(word);
    utt.rate   = 0.85;
    utt.pitch  = 1.0;
    utt.volume = 1.0;
    window.speechSynthesis.speak(utt);
  }, [appendSignTranscript]);

  // ── Reset the word-boundary timer ──────────────────────────────────────────
  const resetWordTimer = useCallback(() => {
    if (wordTimerRef.current) clearTimeout(wordTimerRef.current);
    wordTimerRef.current = setTimeout(() => {
      const word = wordBufferRef.current;
      if (word) {
        speakWord(word);
        wordBufferRef.current = "";
        lastAddedRef.current  = "";
        setSignWordBuffer("");
      }
    }, WORD_TIMEOUT_MS);
  }, [speakWord, setSignWordBuffer]);

  // ── Handle a letter detection result ───────────────────────────────────────
  const onLetterDetected = useCallback((letter: string | null, confidence: number) => {
    if (!letter) {
      // No hand / uncertain — cancel pending hold
      if (holdTimerRef.current) {
        clearTimeout(holdTimerRef.current);
        holdTimerRef.current = null;
      }
      pendingLetterRef.current = null;
      setSignLetter("");
      setSignConfidence(0);
      return;
    }

    setSignLetter(letter);
    setSignConfidence(confidence);

    // If a different letter appears, restart the hold timer
    if (letter !== pendingLetterRef.current) {
      if (holdTimerRef.current) {
        clearTimeout(holdTimerRef.current);
        holdTimerRef.current = null;
      }
      pendingLetterRef.current = letter;

      holdTimerRef.current = setTimeout(() => {
        const held = pendingLetterRef.current;
        // Accept only if the letter is different from the last one added
        if (held && held !== lastAddedRef.current) {
          lastAddedRef.current   = held;
          wordBufferRef.current += held;
          setSignWordBuffer(wordBufferRef.current);
          resetWordTimer();
        }
        holdTimerRef.current = null;
      }, LETTER_HOLD_MS);
    }
  }, [setSignLetter, setSignConfidence, setSignWordBuffer, resetWordTimer]);

  // ── Frame capture loop ──────────────────────────────────────────────────────
  const startCapture = useCallback((video: HTMLVideoElement, ws: WebSocket) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    frameTimerRef.current = window.setInterval(() => {
      if (video.readyState < 2) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      canvas.width  = video.videoWidth  || 320;
      canvas.height = video.videoHeight || 240;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob((blob) => {
        if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob);
      }, "image/jpeg", 0.7);
    }, FRAME_INTERVAL);
  }, []);

  // ── Full cleanup ────────────────────────────────────────────────────────────
  const cleanup = useCallback(() => {
    if (frameTimerRef.current) { clearInterval(frameTimerRef.current); frameTimerRef.current = null; }
    if (wordTimerRef.current)  { clearTimeout(wordTimerRef.current);  wordTimerRef.current  = null; }
    if (holdTimerRef.current)  { clearTimeout(holdTimerRef.current);  holdTimerRef.current  = null; }

    wsRef.current?.close();
    wsRef.current = null;

    if (hiddenVideoRef.current) {
      hiddenVideoRef.current.srcObject = null;
      document.body.removeChild(hiddenVideoRef.current);
      hiddenVideoRef.current = null;
    }
    if (canvasRef.current) {
      document.body.removeChild(canvasRef.current);
      canvasRef.current = null;
    }

    wordBufferRef.current    = "";
    lastAddedRef.current     = "";
    pendingLetterRef.current = null;
    setSignLetter("");
    setSignConfidence(0);
    setSignWordBuffer("");
    setIsConnected(false);
    window.speechSynthesis.cancel();
  }, [setSignLetter, setSignConfidence, setSignWordBuffer]);

  // ── Main effect: connect / disconnect when toggle changes ──────────────────
  useEffect(() => {
    if (!isSignLanguageEnabled) {
      cleanup();
      return;
    }

    // Get the camera MediaStreamTrack from LiveKit
    const pub        = localParticipant?.getTrackPublication(Track.Source.Camera);
    const mediaTrack = pub?.track?.mediaStreamTrack;

    if (!mediaTrack) {
      console.warn("SignLanguage: camera track not available yet");
      return;
    }

    // Create a hidden <video> driven by the existing camera track
    const video       = document.createElement("video");
    video.srcObject   = new MediaStream([mediaTrack]);
    video.autoplay    = true;
    video.muted       = true;
    video.playsInline = true;
    video.style.cssText = "position:fixed;opacity:0;pointer-events:none;width:1px;height:1px;";
    document.body.appendChild(video);
    hiddenVideoRef.current = video;

    // Create a hidden <canvas> for JPEG encoding
    const canvas       = document.createElement("canvas");
    canvas.style.cssText = "position:fixed;opacity:0;pointer-events:none;width:1px;height:1px;";
    document.body.appendChild(canvas);
    canvasRef.current  = canvas;

    // Open WebSocket
    const ws       = new WebSocket(WS_URL);
    ws.binaryType  = "arraybuffer";
    wsRef.current  = ws;

    ws.onopen = () => {
      setIsConnected(true);
      startCapture(video, ws);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data as string) as WsMessage;
        onLetterDetected(data.letter ?? null, data.confidence ?? 0);
      } catch { /* noop */ }
    };

    ws.onclose = () => setIsConnected(false);
    ws.onerror = () => setIsConnected(false);

    return cleanup;
  }, [isSignLanguageEnabled, localParticipant, startCapture, onLetterDetected, cleanup]);

  return { isConnected };
}
