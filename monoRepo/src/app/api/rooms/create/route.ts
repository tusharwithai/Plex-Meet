import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/auth";
import { prisma } from "@/lib/prisma";

export async function POST(req: NextRequest) {
  const session = await auth();

  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const seg = (len: number) =>
    Math.random()
      .toString(36)
      .substring(2, 2 + len)
      .toLowerCase();
  const roomId = `${seg(3)}${seg(4)}${seg(3)}`;

  await prisma.room.upsert({
    where: { id: roomId },
    update: {},
    create: {
      id: roomId,
      hostId: session.user.id,
      isActive: true,
    },
  });

  return NextResponse.json({ roomId });
}
