import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/auth";
import { prisma } from "@/lib/prisma";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ roomId: string }> }
) {
  const { roomId } = await params;
  const session = await auth();

  if (!session?.user?.id) {
    return NextResponse.json({ isHost: false });
  }

  const room = await prisma.room.findUnique({ where: { id: roomId } });

  return NextResponse.json({
    isHost: room?.hostId === session.user.id,
    exists: !!room,
  });
}
