"use client";

import React, { useEffect, useState } from "react";
import { useLocalParticipant } from "@livekit/components-react";
import { Mic, MicOff, Video, VideoOff, Phone, MessageSquare, MonitorUp, SmilePlus, Captions, Info, Users, Copy, Check, Hand, MoreVertical, Lock, ChevronUp } from "lucide-react";
import { useMeetStore } from "@/store/useMeetStore";

const REACTIONS = ["👍", "👎", "👏", "😂", "😮", "😢", "🎉"];

// ── Sign Language icon (ASL hand SVG) ──────────────────────────────────────────
function SignLanguageIcon({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 8h1a4 4 0 0 1 0 8h-1" />
      <path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z" />
      <line x1="6" y1="1" x2="6" y2="8" />
      <line x1="10" y1="1" x2="10" y2="8" />
      <line x1="14" y1="1" x2="14" y2="8" />
    </svg>
  );
}

export default function MeetingControls() {
  const { localParticipant } = useLocalParticipant();
  const {
    isChatOpen, toggleChat,
    isParticipantsOpen, toggleParticipants,
    isCaptionsEnabled, toggleCaptions,
    isHandRaised, toggleHandRaise,
    isSignLanguageEnabled, toggleSignLanguage,
  } = useMeetStore();
  const [time, setTime] = useState("");
  
  // Modals state
  const [showInfo, setShowInfo] = useState(false);
  const [showReactions, setShowReactions] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const update = () => setTime(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, []);

  const toggleMic = () => {
    if (localParticipant) {
      localParticipant.setMicrophoneEnabled(!localParticipant.isMicrophoneEnabled);
    }
  };

  const toggleCam = () => {
    if (localParticipant) {
      localParticipant.setCameraEnabled(!localParticipant.isCameraEnabled);
    }
  };

  const toggleShare = async () => {
    if (localParticipant) {
      try {
        await localParticipant.setScreenShareEnabled(!localParticipant.isScreenShareEnabled);
      } catch (err) {
        console.error("Screen sharing cancelled", err);
      }
    }
  };

  const copyUrl = () => {
    navigator.clipboard.writeText(window.location.href);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const sendReaction = (_emoji: string) => {
    setShowReactions(false);
    // TODO: Broadcast overlay implementation
  };

  const isMicOn  = localParticipant?.isMicrophoneEnabled ?? false;
  const isCamOn  = localParticipant?.isCameraEnabled    ?? false;
  const isSharing = localParticipant?.isScreenShareEnabled ?? false;

  return (
    <div className="flex items-center justify-between px-6 bg-[#202124] text-white z-20 w-full shrink-0 h-20 border-t border-white/10">
      
      {/* Left Info */}
      <div className="flex w-1/4 items-center gap-2">
        <span className="text-sm font-medium">{time}</span>
        <span className="text-white/50">|</span>
        <span className="text-sm font-medium truncate">{typeof window !== 'undefined' ? window.location.pathname.substring(1) : 'Room'}</span>
      </div>

      {/* Center Actions */}
      <div className="flex items-center justify-center gap-2 w-2/4">
        {/* Mic Group */}
        <div className="flex items-center bg-[#3c4043] rounded-full mr-1 overflow-hidden transition-colors">
          <button 
            onClick={toggleMic}
            title="Turn on microphone"
            className={`px-4 py-3 hover:bg-[#4d5156] transition-colors ${!isMicOn ? 'bg-[#ea4335] hover:bg-[#d32f2f]' : ''}`}
          >
            {isMicOn ? <Mic className="w-5 h-5 fill-current" /> : <MicOff className="w-5 h-5" />}
          </button>
          <button className="px-2 border-l border-white/20 hover:bg-[#4d5156] transition-colors h-full flex items-center">
             <ChevronUp className="w-4 h-4" />
          </button>
        </div>

        {/* Cam Group */}
        <div className="flex items-center bg-[#3c4043] rounded-full mr-1 overflow-hidden transition-colors">
          <button 
            onClick={toggleCam}
            title="Turn on camera"
            className={`px-4 py-3 hover:bg-[#4d5156] transition-colors ${!isCamOn ? 'bg-[#ea4335] hover:bg-[#d32f2f]' : ''}`}
          >
            {isCamOn ? <Video className="w-5 h-5 fill-current" /> : <VideoOff className="w-5 h-5" />}
          </button>
          <button className="px-2 border-l border-white/20 hover:bg-[#4d5156] transition-colors h-full flex items-center">
             <ChevronUp className="w-4 h-4" />
          </button>
        </div>

        <button 
          onClick={toggleCaptions}
          title="Turn on captions" 
          className={`p-3 rounded-full transition-colors ${isCaptionsEnabled ? 'bg-[#8ab4f8] text-[#202124]' : 'bg-[#3c4043] hover:bg-[#4d5156]'}`}
        >
          <Captions className={`w-5 h-5 ${isCaptionsEnabled ? 'fill-current' : ''}`} />
        </button>

        {/* ── Sign Language Mode Button ───────────────────────────── */}
        <button
          onClick={toggleSignLanguage}
          title={isSignLanguageEnabled ? "Disable Sign Language Mode" : "Enable Sign Language Mode"}
          className={`p-3 rounded-full transition-all duration-200 relative ${
            isSignLanguageEnabled
              ? 'bg-[#8ab4f8] text-[#202124] shadow-[0_0_12px_rgba(138,180,248,0.5)]'
              : 'bg-[#3c4043] hover:bg-[#4d5156] text-white'
          }`}
        >
          <SignLanguageIcon size={20} />
          {/* Pulsing dot when active */}
          {isSignLanguageEnabled && (
            <span
              style={{
                position: "absolute",
                top: "4px",
                right: "4px",
                width: "7px",
                height: "7px",
                borderRadius: "50%",
                background: "#34d399",
                boxShadow: "0 0 6px #34d399",
                animation: "ping 1.5s cubic-bezier(0,0,0.2,1) infinite",
              }}
            />
          )}
        </button>

        {/* Reactions Wrapper */}
        <div className="relative">
          <button 
            onClick={() => setShowReactions(!showReactions)}
            title="Send a reaction" 
            className={`p-3 rounded-full transition-colors ${showReactions ? 'bg-[#4d5156]' : 'bg-[#3c4043] hover:bg-[#4d5156]'}`}
          >
            <SmilePlus className="w-5 h-5" />
          </button>
          
          {showReactions && (
            <div className="absolute bottom-16 left-1/2 -translate-x-1/2 bg-[#3c4043] border border-white/10 rounded-full flex items-center p-2 shadow-2xl gap-2 animate-in fade-in slide-in-from-bottom-2">
              {REACTIONS.map(emoji => (
                <button 
                  key={emoji}
                  onClick={() => sendReaction(emoji)}
                  className="w-10 h-10 hover:bg-[#4d5156] rounded-full text-xl flex items-center justify-center transition-transform hover:scale-110 active:scale-95"
                >
                  {emoji}
                </button>
              ))}
            </div>
          )}
        </div>

        <button 
          onClick={toggleShare}
          title={isSharing ? "Stop presenting" : "Present now"} 
          className={`p-3 rounded-full transition-colors ${isSharing ? 'bg-[#8ab4f8] text-[#202124]' : 'bg-[#3c4043] hover:bg-[#4d5156]'}`}
        >
          <MonitorUp className="w-5 h-5" />
        </button>

        <button 
          onClick={toggleHandRaise}
          title={isHandRaised ? "Lower hand" : "Raise hand"} 
          className={`p-3 rounded-full transition-colors ${isHandRaised ? 'bg-[#8ab4f8] text-[#202124]' : 'bg-[#3c4043] hover:bg-[#4d5156]'}`}
        >
          <Hand className={`w-5 h-5 ${isHandRaised ? 'fill-current' : ''}`} />
        </button>

        <button 
          title="More options" 
          className={`p-3 rounded-full transition-colors bg-[#3c4043] hover:bg-[#4d5156] mx-1`}
        >
          <MoreVertical className="w-5 h-5" />
        </button>

        <button 
          onClick={() => window.location.href = '/'}
          title="Leave call"
          className="p-3 px-6 bg-[#ea4335] hover:bg-[#d32f2f] text-white rounded-full transition-colors shadow-lg"
        >
          <Phone className="w-5 h-5 transform rotate-[135deg] fill-current" />
        </button>
      </div>

      {/* Right Actions */}
      <div className="flex items-center justify-end w-1/4 gap-1">
        {/* Info Wrapper */}
        <div className="relative">
          <button 
            onClick={() => setShowInfo(!showInfo)}
            title="Meeting details"
            className={`p-3 rounded-full transition-colors ${showInfo ? 'bg-[#4d5156] text-[#8ab4f8]' : 'hover:bg-[#3c4043] text-white'}`}
          >
            <Info className="w-5 h-5" />
          </button>
          
          {showInfo && (
            <div className="absolute bottom-16 right-0 w-80 bg-[#ffffff] text-[#202124] dark:bg-[#202124] dark:text-white border border-border shadow-2xl rounded-xl p-5 animate-in fade-in slide-in-from-bottom-2">
              <h3 className="font-medium text-lg mb-2">Joining info</h3>
              <p className="text-sm mb-4 break-all opacity-80">{window.location.href}</p>
              <button 
                onClick={copyUrl}
                className="w-full bg-[#e8f0fe] hover:bg-[#d2e3fc] dark:bg-[#8ab4f8]/10 dark:hover:bg-[#8ab4f8]/20 text-[#1a73e8] dark:text-[#8ab4f8] flex items-center justify-center gap-2 py-2.5 rounded-md text-sm font-medium transition-colors"
              >
                {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                {copied ? "Link copied" : "Copy joining info"}
              </button>
            </div>
          )}
        </div>

        <button 
          onClick={toggleParticipants}
          title="Show everyone"
          className={`p-3 rounded-full transition-colors ${isParticipantsOpen ? 'bg-[#8ab4f8]/20 text-[#8ab4f8]' : 'hover:bg-[#3c4043] text-white'}`}
        >
           <Users className="w-5 h-5" />
        </button>

        <button 
          onClick={toggleChat}
          title="Chat with everyone"
          className={`p-3 rounded-full transition-colors ${isChatOpen ? 'bg-[#8ab4f8]/20 text-[#8ab4f8]' : 'hover:bg-[#3c4043] text-white'}`}
        >
          <MessageSquare className="w-5 h-5" />
        </button>

        <button 
          title="Host controls"
          className={`p-3 rounded-full transition-colors hover:bg-[#3c4043] text-white`}
        >
          <Lock className="w-5 h-5" />
        </button>
      </div>

      {/* Ping animation for active indicator */}
      <style>{`
        @keyframes ping {
          75%, 100% { transform: scale(2); opacity: 0; }
        }
      `}</style>
    </div>
  );
}
