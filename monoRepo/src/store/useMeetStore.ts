import { create } from 'zustand';

interface MeetStore {
  isMicEnabled: boolean;
  isCamEnabled: boolean;
  isChatOpen: boolean;
  isParticipantsOpen: boolean;
  isCaptionsEnabled: boolean;
  isHandRaised: boolean;
  toggleMic: () => void;
  toggleCam: () => void;
  toggleChat: () => void;
  toggleParticipants: () => void;
  toggleCaptions: () => void;
  toggleHandRaise: () => void;
  closeAllPanels: () => void;

  // ── Sign Language Mode ────────────────────────────────────────────────────
  isSignLanguageEnabled: boolean;
  signLetter: string;
  signConfidence: number;
  signWordBuffer: string;
  signTranscript: string[];
  toggleSignLanguage: () => void;
  setSignLetter: (letter: string) => void;
  setSignConfidence: (confidence: number) => void;
  setSignWordBuffer: (buffer: string) => void;
  appendSignTranscript: (word: string) => void;
  clearSignTranscript: () => void;
}

export const useMeetStore = create<MeetStore>((set) => ({
  isMicEnabled: false,
  isCamEnabled: true,
  isChatOpen: false,
  isParticipantsOpen: false,
  isCaptionsEnabled: false,
  isHandRaised: false,
  toggleMic: () => set((state) => ({ isMicEnabled: !state.isMicEnabled })),
  toggleCam: () => set((state) => ({ isCamEnabled: !state.isCamEnabled })),
  toggleChat: () => set((state) => ({ isChatOpen: !state.isChatOpen, isParticipantsOpen: false })),
  toggleParticipants: () => set((state) => ({ isParticipantsOpen: !state.isParticipantsOpen, isChatOpen: false })),
  toggleCaptions: () => set((state) => ({ isCaptionsEnabled: !state.isCaptionsEnabled })),
  toggleHandRaise: () => set((state) => ({ isHandRaised: !state.isHandRaised })),
  closeAllPanels: () => set({ isChatOpen: false, isParticipantsOpen: false }),

  // ── Sign Language ─────────────────────────────────────────────────────────
  isSignLanguageEnabled: false,
  signLetter: '',
  signConfidence: 0,
  signWordBuffer: '',
  signTranscript: [],
  toggleSignLanguage: () =>
    set((state) => ({
      isSignLanguageEnabled: !state.isSignLanguageEnabled,
      // Reset state when turning off
      ...(!state.isSignLanguageEnabled ? {} : {
        signLetter: '',
        signConfidence: 0,
        signWordBuffer: '',
      }),
    })),
  setSignLetter: (letter) => set({ signLetter: letter }),
  setSignConfidence: (confidence) => set({ signConfidence: confidence }),
  setSignWordBuffer: (buffer) => set({ signWordBuffer: buffer }),
  appendSignTranscript: (word) =>
    set((state) => ({ signTranscript: [...state.signTranscript, word] })),
  clearSignTranscript: () => set({ signTranscript: [], signWordBuffer: '' }),
}));
