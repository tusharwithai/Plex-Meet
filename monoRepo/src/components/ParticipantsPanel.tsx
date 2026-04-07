"use client";

import React from "react";
import { useParticipants } from "@livekit/components-react";
import { X, Mic, MicOff, UserRound } from "lucide-react";
import { useMeetStore } from "@/store/useMeetStore";

export default function ParticipantsPanel() {
  const participants = useParticipants();
  const { toggleParticipants } = useMeetStore();

  return (
    <div className="w-full h-full bg-background border-l border-border flex flex-col pt-4">
      <div className="px-6 flex items-center justify-between pb-4">
        <h2 className="text-foreground font-medium text-lg flex items-center gap-2">
          People <span className="bg-surface text-foreground text-xs font-bold px-2 py-0.5 rounded-full">{participants.length}</span>
        </h2>
        <button onClick={toggleParticipants} className="p-2 hover:bg-surface rounded-full text-muted hover:text-foreground transition-colors">
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-6 mt-2 pb-4">
        {participants.map((p) => (
          <div key={p.identity} className="flex items-center justify-between py-3 group">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-full bg-primary/20 flex flex-shrink-0">
                 <UserRound className="w-5 h-5 m-auto text-primary" />
              </div>
              <div className="flex flex-col">
                <span className="text-sm font-medium text-foreground line-clamp-1 break-all">
                  {p.name || p.identity} {p.isLocal ? "(You)" : ""}
                </span>
                <span className="text-xs text-muted">Meeting host</span>
              </div>
            </div>
            
            <div className="flex-shrink-0 ml-2">
              {p.isMicrophoneEnabled ? (
                <div className="w-8 h-8 rounded-full flex items-center justify-center hover:bg-surface transition-colors">
                  <Mic className="w-4 h-4 text-foreground/70" />
                </div>
              ) : (
                <div className="w-8 h-8 rounded-full flex items-center justify-center bg-danger/10">
                  <MicOff className="w-4 h-4 text-danger" />
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
