"use client";

import React, { useEffect, useState, use } from "react";
import { 
  LiveKitRoom, 
  RoomAudioRenderer,
  GridLayout,
  ParticipantTile,
  useTracks,
  useLocalParticipant
} from "@livekit/components-react";
import { Track } from "livekit-client";
import "@livekit/components-styles";
import { useMeetStore } from "@/store/useMeetStore";
import MeetingControls from "@/components/MeetingControls";
import ChatPanel from "@/components/ChatPanel";
import ParticipantsPanel from "@/components/ParticipantsPanel";
import SignLanguageOverlay from "@/components/SignLanguageOverlay";
import { useSignLanguage } from "@/hooks/useSignLanguage";
import { useSession } from "next-auth/react";

// ── Video Grid ──────────────────────────────────────────────────────────────────
function MyVideoGrid() {
  const tracks = useTracks(
    [
      { source: Track.Source.Camera, withPlaceholder: true },
      { source: Track.Source.ScreenShare, withPlaceholder: false },
    ],
    { onlySubscribed: false },
  );

  const localTracks  = tracks.filter(t => t.participant.isLocal);
  const remoteTracks = tracks.filter(t => !t.participant.isLocal);
  const primaryLocal = localTracks.find(t => t.source === Track.Source.Camera) || localTracks[0];

  // 1 Person: Centered
  if (remoteTracks.length === 0) {
    return (
      <div className="w-full h-full p-4 flex items-center justify-center">
        <div className="w-full max-w-5xl aspect-video rounded-xl overflow-hidden bg-[#3c4043] shadow-md border border-white/5 relative">
          {primaryLocal && <ParticipantTile trackRef={primaryLocal} className="!w-full !h-full absolute inset-0 [&>video]:!object-cover" />}
        </div>
      </div>
    );
  }

  // 2 Persons (1-on-1): Focus remote, PIP local
  if (remoteTracks.length === 1) {
    return (
      <div className="w-full h-full p-4 flex items-center justify-center">
        <div className="w-full h-full relative bg-[#3c4043] rounded-xl overflow-hidden border border-white/5 shadow-md">
          <div className="absolute inset-0 overflow-hidden">
            <ParticipantTile trackRef={remoteTracks[0]} className="!w-full !h-full" />
          </div>
          {primaryLocal && (
            <div className="absolute bottom-6 right-6 w-72 aspect-video rounded-xl overflow-hidden shadow-2xl border border-white/20 z-10 bg-[#202124] transition-all hover:scale-105 cursor-pointer">
               <ParticipantTile trackRef={primaryLocal} className="!w-full !h-full absolute inset-0 [&>video]:!object-cover" />
            </div>
          )}
        </div>
      </div>
    );
  }

  // 3+ Persons: Grid remotes, PIP local
  return (
    <div className="p-4 w-full h-full relative flex items-center justify-center">
      <div className="w-full h-full relative bg-[#3c4043] rounded-xl overflow-hidden shadow-md border border-white/5">
        <GridLayout tracks={remoteTracks} className="!w-full !h-full">
          <ParticipantTile />
        </GridLayout>
      </div>
      {primaryLocal && (
        <div className="absolute bottom-10 right-10 w-64 aspect-video rounded-xl overflow-hidden shadow-2xl border border-white/20 z-10 bg-[#202124] transition-all hover:scale-105 cursor-pointer">
           <ParticipantTile trackRef={primaryLocal} className="!w-full !h-full absolute inset-0 [&>video]:!object-cover" />
        </div>
      )}
    </div>
  );
}

// ── Sign Language Bridge (must be inside LiveKitRoom context) ───────────────────
function SignLanguageBridge() {
  const { localParticipant } = useLocalParticipant();
  const { isSignLanguageEnabled } = useMeetStore();
  const { isConnected } = useSignLanguage(localParticipant);

  if (!isSignLanguageEnabled) return null;

  return (
    <div
      style={{
        position: "absolute",
        bottom: "96px",      // sits just above the 80px control bar
        left: "20px",
        zIndex: 40,
        animation: "slideInOverlay 0.25s ease-out",
      }}
    >
      <SignLanguageOverlay isConnected={isConnected} />
      <style>{`
        @keyframes slideInOverlay {
          from { opacity: 0; transform: translateY(12px); }
          to   { opacity: 1; transform: translateY(0);    }
        }
      `}</style>
    </div>
  );
}

// ── Main Room Page ──────────────────────────────────────────────────────────────
export default function MeetingRoom({ params }: { params: Promise<{ roomId: string }> }) {
  const unwrappedParams = use(params);
  const roomId = unwrappedParams.roomId;
  const [token, setToken] = useState("");
  const { isCamEnabled, isMicEnabled, isChatOpen, isParticipantsOpen } = useMeetStore();
  const { data: session } = useSession();
  
  const serverUrl = process.env.NEXT_PUBLIC_LIVEKIT_URL || "wss://test-project.livekit.cloud";
  const participantName = session?.user?.name ?? session?.user?.email ?? `Guest-${Math.floor(Math.random() * 9999)}`;

  useEffect(() => {
    const getToken = async () => {
      try {
        const res  = await fetch(`/api/token?roomName=${roomId}&participantName=${participantName}`);
        const data = await res.json();
        if (data.token) {
          setToken(data.token);
        } else {
          console.error("No token received", data);
        }
      } catch (e) {
        console.error("Failed to fetch token", e);
      }
    };
    getToken();
  }, [roomId, participantName]);

  const isFallbackUrl = serverUrl === "wss://test-project.livekit.cloud";

  if (isFallbackUrl) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-background text-foreground p-6 text-center">
        <div className="w-16 h-16 bg-danger/20 text-danger rounded-full flex items-center justify-center mb-6">
          <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <h1 className="text-3xl font-medium mb-4">LiveKit Not Configured</h1>
        <p className="text-muted max-w-lg mb-8">
          You are seeing the &quot;Disconnected&quot; error because the app is using the dummy fallback URL. To securely connect the video grid, please configure your <strong>.env</strong> files!
        </p>
        <div className="text-left bg-surface p-6 rounded-lg border border-border w-full max-w-2xl font-mono text-sm overflow-x-auto">
          <p className="text-primary mb-2">1. In frontend/.env.local (Create if missing):</p>
          <code className="block text-foreground mb-6">NEXT_PUBLIC_LIVEKIT_URL=wss://your-project.livekit.cloud</code>
          <p className="text-primary mb-2">2. In backend/.env:</p>
          <code className="block text-foreground">LIVEKIT_API_KEY=your_api_key<br/>LIVEKIT_API_SECRET=your_api_secret</code>
        </div>
        <button 
          onClick={() => window.location.reload()}
          className="mt-8 bg-primary hover:bg-primary-hover text-white dark:text-gray-900 px-6 py-2 rounded-md font-medium transition-colors"
        >
          I have added them, retry
        </button>
      </div>
    );
  }

  if (token === "") {
    return (
      <div className="flex items-center justify-center min-h-screen bg-[#202124] text-white">
        <p className="text-xl">Joining meeting...</p>
      </div>
    );
  }

  return (
    <LiveKitRoom
      video={isCamEnabled}
      audio={isMicEnabled}
      token={token}
      serverUrl={serverUrl}
      data-lk-theme="default"
      className="bg-[#202124] text-white"
      style={{ width: '100%', height: '100dvh', position: 'relative', overflow: 'hidden' }}
    >
      {/* Main Content Area (Videos + side panels) */}
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: '80px', display: 'flex', overflow: 'hidden' }}>
        <div className="flex-1 w-full h-full overflow-hidden transition-all duration-300 ease-in-out relative">
           <MyVideoGrid />
        </div>
        
        {isChatOpen && (
           <div className="w-80 border-l border-white/10 h-full bg-[#202124] z-10 flex-shrink-0 animate-in slide-in-from-right-10 duration-200">
              <ChatPanel roomId={roomId} />
           </div>
        )}
        
        {isParticipantsOpen && (
           <div className="w-80 border-l border-white/10 h-full bg-[#202124] z-10 flex-shrink-0 animate-in slide-in-from-right-10 duration-200">
              <ParticipantsPanel />
           </div>
        )}
      </div>
      
      {/* Sign Language Overlay + Hook (inside LiveKitRoom context) */}
      <SignLanguageBridge />

      {/* Bottom Controls */}
      <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: '80px', zIndex: 50 }}>
        <MeetingControls />
      </div>

      <RoomAudioRenderer />
    </LiveKitRoom>
  );
}
