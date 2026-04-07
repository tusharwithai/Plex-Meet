import { AccessToken } from 'livekit-server-sdk';
import { NextRequest, NextResponse } from 'next/server';

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const roomName = searchParams.get('roomName');
  const participantName = searchParams.get('participantName');

  if (!roomName || !participantName) {
    return NextResponse.json({ error: 'Missing roomName or participantName' }, { status: 400 });
  }

  const apiKey = process.env.LIVEKIT_API_KEY;
  const apiSecret = process.env.LIVEKIT_API_SECRET;

  if (!apiKey || !apiSecret) {
    return NextResponse.json({ error: 'Server misconfigured' }, { status: 500 });
  }

  try {
    const at = new AccessToken(apiKey, apiSecret, {
      identity: participantName,
      name: participantName,
    });

    at.addGrant({ roomJoin: true, room: roomName });
    
    // Using await here because toJwt might be async in some server-sdk versions, 
    // or just to be safe.
    const token = await at.toJwt();
    
    return NextResponse.json({ token });
  } catch (error) {
    console.error("Token generation failed:", error);
    return NextResponse.json({ error: "Failed to generate token" }, { status: 500 });
  }
}
