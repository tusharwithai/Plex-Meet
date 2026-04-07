"use client";

import React, { useEffect, useRef, useState, use, useCallback } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Mic, MicOff, Video, VideoOff, MoreVertical, Sun, Moon, Clock, CheckCircle, XCircle } from "lucide-react";
import { useMeetStore } from "@/store/useMeetStore";
import { useTheme } from "next-themes";
import { useSession } from "next-auth/react";
import type { Socket } from "socket.io-client";

interface WaitingUser {
  socketId: string;
  name: string;
}

export default function LobbyPage({ params }: { params: Promise<{ roomId: string }> }) {
  const unwrappedParams = use(params);
  const roomId = unwrappedParams.roomId;
  const router = useRouter();
  const searchParams = useSearchParams();
  const { resolvedTheme, setTheme } = useTheme();
  const { isMicEnabled, isCamEnabled, toggleMic: toggleStoreMic, toggleCam: toggleStoreCam } = useMeetStore();
  const { data: session, status } = useSession();

  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [errorDesc, setErrorDesc] = useState<string | null>(null);

  const [isHost, setIsHost] = useState(false);
  const [isCheckingHost, setIsCheckingHost] = useState(true);
  const [waitingStatus, setWaitingStatus] = useState<"idle" | "waiting" | "admitted" | "denied">("idle");
  const [waitingUsers, setWaitingUsers] = useState<WaitingUser[]>([]);

  // Single shared socket ref
  const socketRef = useRef<Socket | null>(null);

  // ── Determine if the user is the host ───────────────────────────────────────
  useEffect(() => {
    if (status === "loading") return;

    const checkHost = async () => {
      try {
        const res = await fetch(`/api/rooms/${roomId}/check-host`);
        const data = await res.json();
        setIsHost(data.isHost === true);
      } catch {
        setIsHost(false);
      } finally {
        setIsCheckingHost(false);
      }
    };

    checkHost();
  }, [roomId, status, session]);

  // ── Camera / Microphone setup ────────────────────────────────────────────────
  useEffect(() => {
    let activeStream: MediaStream | null = null;
    const startMedia = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          setErrorDesc("Media devices not supported in this browser.");
          return;
        }

        const constraints = {
          video: isCamEnabled ? { facingMode: "user" } : false,
          audio: isMicEnabled,
        };

        if (!constraints.video && !constraints.audio) {
          if (videoRef.current) videoRef.current.srcObject = null;
          return;
        }

        activeStream = await navigator.mediaDevices.getUserMedia(constraints);
        setStream(activeStream);

        if (videoRef.current && isCamEnabled) {
          videoRef.current.srcObject = activeStream;
        }
        setErrorDesc(null);
      } catch (err: unknown) {
        console.error("Error accessing media:", err);
        setErrorDesc("Cannot access camera or microphone. Please check permissions.");
      }
    };

    startMedia();

    return () => {
      if (activeStream) activeStream.getTracks().forEach((t) => t.stop());
    };
  }, [isCamEnabled, isMicEnabled]);

  // ── Socket.io: connect once host-check is done ───────────────────────────────
  useEffect(() => {
    if (isCheckingHost) return;

    const initSocket = async () => {
      const { io } = await import("socket.io-client");
      const BACKEND_URL = process.env.NEXT_PUBLIC_SIGNALING_URL || "http://localhost:3001";

      socketRef.current = io(BACKEND_URL, { transports: ["websocket"] });

      socketRef.current.on("connect", () => {
        if (isHost) {
          socketRef.current!.emit("host-join", { roomId, hostId: session?.user?.id });
        }
      });

      if (isHost) {
        socketRef.current.on("user-requesting-join", (data: WaitingUser) => {
          setWaitingUsers((prev) =>
            prev.find((u) => u.socketId === data.socketId) ? prev : [...prev, data]
          );
        });
      } else {
        socketRef.current.on("admitted", () => {
          setWaitingStatus("admitted");
          setTimeout(() => router.push(`/${roomId}`), 600);
        });

        socketRef.current.on("denied", () => {
          setWaitingStatus("denied");
        });
      }
    };

    initSocket();

    return () => {
      socketRef.current?.disconnect();
      socketRef.current = null;
    };
  }, [isCheckingHost, isHost, roomId, session, router]);

  // ── Controls ─────────────────────────────────────────────────────────────────
  const toggleMic = () => {
    if (stream) stream.getAudioTracks().forEach((t) => (t.enabled = !isMicEnabled));
    toggleStoreMic();
  };

  const toggleCam = () => {
    if (stream) stream.getVideoTracks().forEach((t) => (t.enabled = !isCamEnabled));
    toggleStoreCam();
  };

  const handleJoin = useCallback(() => {
    if (isHost) {
      router.push(`/${roomId}`);
      return;
    }

    // Guest: emit request-join on the existing socket
    if (socketRef.current?.connected) {
      const guestName = session?.user?.name ?? `Guest-${Math.floor(Math.random() * 9999)}`;
      socketRef.current.emit("request-join", { roomId, name: guestName });
      setWaitingStatus("waiting");
    } else {
      // Socket not connected yet — retry shortly
      setTimeout(handleJoin, 500);
    }
  }, [isHost, roomId, router, session]);

  const handleAdmit = (socketId: string) => {
    fetch(`/api/rooms/${roomId}/admit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ socketId }),
    });
    setWaitingUsers((prev) => prev.filter((u) => u.socketId !== socketId));
  };

  const handleDeny = (socketId: string) => {
    fetch(`/api/rooms/${roomId}/deny`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ socketId }),
    });
    setWaitingUsers((prev) => prev.filter((u) => u.socketId !== socketId));
  };

  // ── Loading state ────────────────────────────────────────────────────────────
  if (isCheckingHost) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          <p className="text-muted text-lg">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col min-h-screen bg-background text-foreground transition-colors">
      {/* Lobby Header */}
      <header className="p-4 px-6 flex items-center justify-between border-b border-border">
        <div className="text-xl font-medium tracking-tight">Google Meet</div>
        <div className="flex items-center gap-4">
          <button
            suppressHydrationWarning
            onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
            className="w-10 h-10 flex items-center justify-center hover:bg-surface rounded-full transition-colors text-muted hover:text-foreground"
            aria-label="Toggle Dark Mode"
          >
            {resolvedTheme === "dark" ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
          </button>
        </div>
      </header>

      {/* Lobby Body */}
      <main className="flex-1 flex flex-col md:flex-row items-center justify-center p-6 gap-8 max-w-6xl mx-auto w-full">
        {/* Left: Video Preview */}
        <div className="flex-1 flex flex-col items-center justify-center relative w-full aspect-video md:max-w-[700px] bg-surface rounded-lg overflow-hidden shadow-lg border border-border mt-4 md:mt-0">
          {errorDesc ? (
            <div className="flex flex-col items-center justify-center w-full h-full text-center p-4 gap-3">
              <VideoOff className="w-10 h-10 text-muted" />
              <p className="text-foreground text-lg">{errorDesc}</p>
            </div>
          ) : isCamEnabled ? (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover transform scale-x-[-1]"
            />
          ) : (
            <div className="flex flex-col items-center justify-center w-full h-full text-center">
              <div className="w-24 h-24 rounded-full bg-purple-600 flex items-center justify-center text-white text-4xl mb-6 shadow-md">
                {session?.user?.name?.[0]?.toUpperCase() ?? "G"}
              </div>
              <p className="absolute bottom-6 text-white bg-black/60 px-3 py-1.5 rounded-md text-sm backdrop-blur-sm tracking-wide">
                Camera is off
              </p>
            </div>
          )}

          {/* Controls overlay */}
          <div className="absolute bottom-6 left-0 right-0 flex justify-center gap-4">
            <button
              onClick={toggleMic}
              className={`p-4 rounded-full transition-colors ${isMicEnabled ? "bg-surface hover:bg-surface-hover text-foreground" : "bg-danger text-white"}`}
            >
              {isMicEnabled ? <Mic className="w-5 h-5" /> : <MicOff className="w-5 h-5" />}
            </button>
            <button
              onClick={toggleCam}
              className={`p-4 rounded-full transition-colors ${isCamEnabled ? "bg-surface hover:bg-surface-hover text-foreground" : "bg-danger text-white"}`}
            >
              {isCamEnabled ? <Video className="w-5 h-5" /> : <VideoOff className="w-5 h-5" />}
            </button>
            <div className="w-px h-10 bg-border self-center mx-2 hidden md:block" />
            <button className="p-4 rounded-full bg-surface/80 hover:bg-surface text-foreground border border-border backdrop-blur-md transition-colors shadow-lg">
              <MoreVertical className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Right: Join / Status Panel */}
        <div className="w-full max-w-[400px] flex flex-col items-center text-center p-4 md:p-8 animate-fade-in">
          <h2 className="text-3xl font-normal mb-2 text-foreground">
            {isHost ? "Your meeting is ready" : "Ready to join?"}
          </h2>

          {isHost && (
            <div className="flex items-center gap-2 mb-6 mt-1">
              <span className="text-sm text-muted">Room code:</span>
              <code className="text-sm font-mono bg-surface px-2 py-1 rounded border border-border text-foreground">
                {roomId}
              </code>
            </div>
          )}

          {/* Waiting status for guests */}
          {waitingStatus === "waiting" && (
            <div className="flex flex-col items-center gap-3 mb-6 p-4 bg-surface rounded-lg border border-border w-full">
              <Clock className="w-8 h-8 text-primary animate-pulse" />
              <p className="text-foreground font-medium">Waiting for host to admit you…</p>
              <p className="text-muted text-sm">The host has been notified of your request.</p>
            </div>
          )}

          {waitingStatus === "admitted" && (
            <div className="flex flex-col items-center gap-3 mb-6 p-4 bg-green-500/10 rounded-lg border border-green-500/30 w-full">
              <CheckCircle className="w-8 h-8 text-green-500" />
              <p className="text-foreground font-medium">Admitted! Joining now…</p>
            </div>
          )}

          {waitingStatus === "denied" && (
            <div className="flex flex-col items-center gap-3 mb-6 p-4 bg-red-500/10 rounded-lg border border-red-500/30 w-full">
              <XCircle className="w-8 h-8 text-red-500" />
              <p className="text-foreground font-medium">Your request was denied.</p>
              <p className="text-muted text-sm">The host declined your request to join.</p>
              <button
                onClick={() => router.push("/")}
                className="mt-2 text-primary text-sm hover:underline"
              >
                Return to home
              </button>
            </div>
          )}

          {/* Host: waiting guests panel */}
          {isHost && waitingUsers.length > 0 && (
            <div className="w-full mb-6 p-4 bg-surface rounded-lg border border-border">
              <p className="text-sm font-medium text-foreground mb-3 text-left">
                Waiting to join ({waitingUsers.length})
              </p>
              <div className="flex flex-col gap-2">
                {waitingUsers.map((u) => (
                  <div key={u.socketId} className="flex items-center justify-between p-2 rounded bg-background">
                    <div className="flex items-center gap-2">
                      <div className="w-7 h-7 rounded-full bg-purple-600 flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
                        {u.name[0].toUpperCase()}
                      </div>
                      <span className="text-sm text-foreground truncate max-w-[120px]">{u.name}</span>
                    </div>
                    <div className="flex gap-2 flex-shrink-0">
                      <button
                        onClick={() => handleAdmit(u.socketId)}
                        className="text-xs px-3 py-1 bg-green-600 hover:bg-green-700 text-white rounded-full transition-colors"
                      >
                        Admit
                      </button>
                      <button
                        onClick={() => handleDeny(u.socketId)}
                        className="text-xs px-3 py-1 bg-transparent hover:bg-red-600/20 text-red-500 border border-red-500/40 rounded-full transition-colors"
                      >
                        Deny
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Join / Enter button */}
          {waitingStatus === "idle" && (
            <div className="flex flex-col sm:flex-row items-center gap-4 w-full justify-center">
              <button
                onClick={handleJoin}
                className="bg-primary hover:bg-primary-hover text-white dark:text-gray-900 font-medium px-8 py-3 rounded-full flex-1 w-full flex items-center justify-center transition-colors shadow-sm text-base"
              >
                {isHost ? "Start meeting" : "Ask to join"}
              </button>
            </div>
          )}

          {!isHost && waitingStatus === "idle" && (
            <p className="mt-4 text-sm text-muted">
              {session?.user
                ? `Joining as ${session.user.name}`
                : "You are joining as an anonymous guest"}
            </p>
          )}
        </div>
      </main>
    </div>
  );
}
