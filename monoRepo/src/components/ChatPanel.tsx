"use client";

import React, { useEffect, useState, useRef, useCallback } from "react";
import { Send, X } from "lucide-react";
import { useMeetStore } from "@/store/useMeetStore";
import {
  useLocalParticipant,
  useDataChannel,
} from "@livekit/components-react";

interface ChatMessage {
  sender: string;
  text: string;
  time: string;
}

const CHAT_TOPIC = "chat";

export default function ChatPanel({ roomId: _roomId }: { roomId: string }) {
  const { toggleChat } = useMeetStore();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { localParticipant } = useLocalParticipant();

  const myName = localParticipant?.name || localParticipant?.identity || "Me";

  // Receive messages via LiveKit Data Channel
  const onDataReceived = useCallback(
    (msg: { payload: Uint8Array; topic?: string }) => {
      if (msg.topic !== CHAT_TOPIC) return;
      try {
        const decoded = new TextDecoder().decode(msg.payload);
        const data: ChatMessage = JSON.parse(decoded);
        setMessages((prev) => [...prev, data]);
      } catch (e) {
        console.error("Failed to decode chat message", e);
      }
    },
    []
  );

  const { send } = useDataChannel(CHAT_TOPIC, onDataReceived);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !send) return;

    const data: ChatMessage = {
      sender: myName,
      text: input.trim(),
      time: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };

    // Optimistically add own message to local state
    setMessages((prev) => [...prev, data]);

    // Send to all remote participants
    const encoded = new TextEncoder().encode(JSON.stringify(data));
    send(encoded, { reliable: true });

    setInput("");
  };

  return (
    <div className="w-full h-full bg-background border-l border-border flex flex-col pt-4">
      <div className="px-6 flex items-center justify-between pb-4">
        <h2 className="text-foreground font-medium text-lg">In-call messages</h2>
        <button
          onClick={toggleChat}
          className="p-2 hover:bg-surface rounded-full text-muted hover:text-foreground transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="bg-surface/50 text-xs text-center py-2 text-muted px-4 mb-2">
        Messages can only be seen by people in the call and are deleted when the call ends.
      </div>

      <div className="flex-1 overflow-y-auto px-6 space-y-4 pb-4">
        {messages.map((msg, i) => {
          const isMe = msg.sender === myName;
          return (
            <div key={i} className={`flex flex-col ${isMe ? "items-end" : "items-start"}`}>
              <div className="flex items-baseline gap-2 mb-1">
                <span className="text-xs font-semibold text-foreground">
                  {isMe ? "You" : msg.sender}
                </span>
                <span className="text-[10px] text-muted">{msg.time}</span>
              </div>
              <span
                className={`text-sm p-3 rounded-xl w-max max-w-[90%] shadow-sm ${
                  isMe
                    ? "bg-primary/20 text-primary rounded-tr-none"
                    : "bg-surface text-foreground rounded-tl-none"
                }`}
              >
                {msg.text}
              </span>
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 bg-background">
        <form
          onSubmit={sendMessage}
          className="relative flex items-center bg-surface rounded-full border border-border focus-within:border-primary focus-within:ring-1 focus-within:ring-primary overflow-hidden pr-2"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Send a message"
            className="flex-1 bg-transparent border-none px-4 py-3 text-sm focus:outline-none text-foreground placeholder-muted"
          />
          <button
            type="submit"
            disabled={!input.trim()}
            className="p-2 text-primary hover:bg-primary/10 rounded-full disabled:opacity-50 transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>
    </div>
  );
}
