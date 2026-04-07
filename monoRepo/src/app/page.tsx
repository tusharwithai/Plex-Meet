"use client";

import React, { useState, useEffect } from "react";
import { Video, Keyboard, Settings, HelpCircle, MessageSquare, Moon, Sun, LogOut } from "lucide-react";
import { useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import { useSession, signIn, signOut } from "next-auth/react";

export default function Home() {
  const [currentTime, setCurrentTime] = useState<string>("");
  const [currentDate, setCurrentDate] = useState<string>("");
  const [meetingCode, setMeetingCode] = useState<string>("");
  const [isCreating, setIsCreating] = useState(false);
  const router = useRouter();
  const { resolvedTheme, setTheme } = useTheme();
  const { data: session, status } = useSession();

  useEffect(() => {
    const updateDateTime = () => {
      const now = new Date();
      setCurrentTime(now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
      setCurrentDate(now.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" }));
    };
    updateDateTime();
    const interval = setInterval(updateDateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleJoin = () => {
    if (meetingCode.trim()) {
      const roomId = meetingCode.replace(/-/g, "");
      router.push(`/${roomId}/lobby`);
    }
  };

  const handleNewMeeting = async () => {
    if (!session?.user) {
      signIn("google");
      return;
    }
    setIsCreating(true);
    try {
      const res = await fetch("/api/rooms/create", { method: "POST" });
      const data = await res.json();
      if (data.roomId) {
        router.push(`/${data.roomId}/lobby?host=true`);
      }
    } finally {
      setIsCreating(false);
    }
  };

  const userInitial = session?.user?.name?.[0]?.toUpperCase() ?? "?";

  return (
    <div className="flex flex-col min-h-screen">
      {/* Header */}
      <header className="flex items-center justify-between p-4 px-6 bg-background">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center text-white">
            <Video className="w-5 h-5" />
          </div>
          <span className="text-xl font-medium tracking-tight text-foreground">Google Meet</span>
        </div>

        <div className="flex items-center gap-6 text-muted">
          <span className="text-sm font-medium hidden md:block">
            {currentTime} • {currentDate}
          </span>
          <div className="flex items-center gap-2">
            <HelpCircle className="w-10 h-10 cursor-pointer hover:bg-surface rounded-full p-2.5 transition-colors" />
            <MessageSquare className="w-10 h-10 cursor-pointer hover:bg-surface rounded-full p-2.5 transition-colors" />
            <Settings className="w-10 h-10 cursor-pointer hover:bg-surface rounded-full p-2.5 transition-colors" />

            {/* Theme Toggle */}
            <button
              suppressHydrationWarning
              onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
              className="w-10 h-10 flex items-center justify-center hover:bg-surface rounded-full transition-colors ml-2"
              aria-label="Toggle Dark Mode"
            >
              {resolvedTheme === "dark" ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
            </button>
          </div>

          {/* Auth Area */}
          {status === "authenticated" ? (
            <div className="flex items-center gap-2">
              {session.user?.image ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={session.user.image}
                  alt={session.user.name ?? "User"}
                  className="w-8 h-8 rounded-full cursor-pointer border border-border"
                  title={session.user.name ?? ""}
                />
              ) : (
                <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center text-white text-sm font-semibold cursor-pointer">
                  {userInitial}
                </div>
              )}
              <button
                onClick={() => signOut()}
                title="Sign out"
                className="w-8 h-8 flex items-center justify-center hover:bg-surface rounded-full transition-colors text-muted hover:text-foreground"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <button
              onClick={() => signIn("google")}
              className="text-sm font-medium px-4 py-2 bg-primary text-white rounded-md hover:bg-primary-hover transition-colors"
            >
              Sign in
            </button>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex flex-col md:flex-row items-center justify-center p-6 md:p-12 gap-12 lg:gap-24 relative overflow-hidden bg-background">
        {/* Left Side: Call to actions */}
        <div className="flex-1 max-w-xl flex flex-col justify-center animate-fade-in z-10">
          <h1 className="text-4xl md:text-[2.75rem] leading-[1.2] font-normal tracking-tight text-foreground mb-4">
            Premium video meetings. Now free for everyone.
          </h1>
          <p className="text-muted text-lg mb-8 max-w-md">
            We re-engineered the service we built for secure business meetings, Google Meet, to make it free and
            available for all.
          </p>

          <div className="flex flex-wrap items-center gap-4">
            <button
              onClick={handleNewMeeting}
              disabled={isCreating}
              className="bg-primary hover:bg-primary-hover text-white dark:text-gray-900 font-medium px-6 py-3 rounded-md flex items-center gap-2 transition-colors whitespace-nowrap disabled:opacity-70 disabled:cursor-not-allowed"
            >
              <Video className="w-5 h-5" />
              {isCreating ? "Creating..." : "New meeting"}
            </button>
            <div className="flex items-center gap-2 h-12 w-full sm:w-auto relative group">
              <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none text-muted">
                <Keyboard className="w-5 h-5" />
              </div>
              <input
                type="text"
                placeholder="Enter a code or link"
                value={meetingCode}
                onChange={(e) => setMeetingCode(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleJoin()}
                className="pl-10 pr-4 py-3 bg-transparent border border-border rounded-md text-foreground placeholder-muted focus:border-primary focus:bg-surface outline-none transition-colors w-full sm:w-64"
              />
            </div>
            {meetingCode.trim().length > 0 && (
              <button
                onClick={handleJoin}
                className="text-primary font-medium px-4 py-2 hover:bg-primary/10 rounded-md transition-colors"
              >
                Join
              </button>
            )}
          </div>
          <div className="mt-8 border-t border-border pt-6">
            <p className="text-muted text-sm">
              <a href="#" className="text-primary hover:underline">
                Learn more
              </a>{" "}
              about Google Meet
            </p>
          </div>
        </div>

        {/* Right Side: Hero Illustration */}
        <div className="flex-1 w-full max-w-md hidden md:flex flex-col items-center text-center">
          <div className="w-72 h-72 rounded-full border border-border mb-6 flex items-center justify-center shadow-md bg-gradient-to-tr from-background to-surface">
            <div className="grid grid-cols-2 gap-2 w-32 h-32 opacity-90">
              <div className="bg-primary rounded-md"></div>
              <div className="bg-[#ea4335] rounded-md"></div>
              <div className="bg-[#fbbc04] rounded-md"></div>
              <div className="bg-[#34a853] rounded-md"></div>
            </div>
          </div>
          <h2 className="text-xl font-medium text-foreground mb-2">Get a link that you can share</h2>
          <p className="text-muted text-sm max-w-xs">
            Click <strong>New meeting</strong> to get a link that you can send to people that you want to meet with
          </p>
        </div>
      </main>
    </div>
  );
}
