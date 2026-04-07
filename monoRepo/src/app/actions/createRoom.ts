"use server";

import { auth } from "@/auth";
import { prisma } from "@/lib/prisma";
import { redirect } from "next/navigation";

/**
 * Creates a new room owned by the current user, registers it in the DB,
 * and redirects the host directly into the meeting room (bypassing waiting room).
 */
export async function createRoom() {
  const session = await auth();

  if (!session?.user?.id) {
    redirect("/");
  }

  // Generate a Google-Meet-style room ID: xxx-yyyy-xxx
  const seg = (len: number) =>
    Math.random()
      .toString(36)
      .substring(2, 2 + len)
      .toLowerCase();
  const roomId = `${seg(3)}${seg(4)}${seg(3)}`;

  // Upsert the room so refreshes don't create duplicates
  await prisma.room.upsert({
    where: { id: roomId },
    update: {},
    create: {
      id: roomId,
      hostId: session.user.id,
      isActive: true,
    },
  });

  redirect(`/${roomId}/lobby?host=true`);
}
