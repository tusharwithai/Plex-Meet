import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/auth";

const SIGNALING_URL = process.env.SIGNALING_INTERNAL_URL || "http://localhost:3001";

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ roomId: string }> }
) {
  const { roomId } = await params;
  const session = await auth();

  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { socketId } = await req.json();

  try {
    await fetch(`${SIGNALING_URL}/admin/deny`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ roomId, socketId }),
    });
  } catch (e) {
    console.error("Failed to signal socket server:", e);
  }

  return NextResponse.json({ ok: true });
}
