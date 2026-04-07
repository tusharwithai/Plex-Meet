/**
 * Sign-Meet Signaling Server
 * Handles Socket.io for: chat, waiting room (request-join, admit-guest, deny-guest)
 *
 * Start: node server.js
 * Env:  PORT=3001 (default)
 */

const express = require("express");
const http = require("http");
const { Server } = require("socket.io");

const app = express();
app.use(express.json());

const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"],
  },
});

// ─────────────────────────────────────────────────────────────────────────────
// State
// hostSockets[roomId] = socketId of the host socket
// waitingGuests[socketId] = { roomId, name }
// ─────────────────────────────────────────────────────────────────────────────
const hostSockets = {}; // roomId → host socketId
const waitingGuests = {}; // guestSocketId → { roomId, name }

// ─────────────────────────────────────────────────────────────────────────────
// Socket.io events
// ─────────────────────────────────────────────────────────────────────────────
io.on("connection", (socket) => {
  console.log(`[socket] connected: ${socket.id}`);

  // ── Host announces themselves ──────────────────────────────────────────────
  socket.on("host-join", ({ roomId, hostId }) => {
    console.log(`[host] ${socket.id} is host of room ${roomId}`);
    hostSockets[roomId] = socket.id;
    socket.join(`room:${roomId}`);
  });

  // ── Guest requests to join ─────────────────────────────────────────────────
  socket.on("request-join", ({ roomId, name }) => {
    console.log(`[guest] ${socket.id} (${name}) wants to join ${roomId}`);
    waitingGuests[socket.id] = { roomId, name };
    socket.join(`room:${roomId}`);

    // Notify the host
    const hostSocketId = hostSockets[roomId];
    if (hostSocketId) {
      io.to(hostSocketId).emit("user-requesting-join", {
        socketId: socket.id,
        name,
      });
    } else {
      // No host in room yet — deny immediately
      socket.emit("denied");
    }
  });

  // ── Chat messages ──────────────────────────────────────────────────────────
  socket.on("send-message", ({ roomId, message, sender }) => {
    io.to(`room:${roomId}`).emit("receive-message", { sender, message });
  });

  // ── Cleanup on disconnect ──────────────────────────────────────────────────
  socket.on("disconnect", () => {
    console.log(`[socket] disconnected: ${socket.id}`);

    // Remove from host registry
    for (const [roomId, hostId] of Object.entries(hostSockets)) {
      if (hostId === socket.id) {
        delete hostSockets[roomId];
        break;
      }
    }

    delete waitingGuests[socket.id];
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Admin HTTP endpoints (called by Next.js API routes)
// ─────────────────────────────────────────────────────────────────────────────
app.post("/admin/admit", (req, res) => {
  const { roomId, socketId } = req.body;
  console.log(`[admin] admitting ${socketId} to ${roomId}`);
  io.to(socketId).emit("admitted");
  delete waitingGuests[socketId];
  res.json({ ok: true });
});

app.post("/admin/deny", (req, res) => {
  const { roomId, socketId } = req.body;
  console.log(`[admin] denying ${socketId} from ${roomId}`);
  io.to(socketId).emit("denied");
  delete waitingGuests[socketId];
  res.json({ ok: true });
});

// Health check
app.get("/health", (req, res) => res.json({ status: "ok" }));

// ─────────────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  console.log(`[server] Signaling server running on http://localhost:${PORT}`);
});
