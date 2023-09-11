const http = require('http');
const { Server } = require('socket.io');

const httpServer = http.createServer();
const io = new Server(httpServer, {
  cors: {
    origin: '*',
  },
});

io.on('connection', (socket) => {
  console.log('Client connected');

  // Add this event listener
  socket.on('dataUpdate', (updatedData) => {
    // Forward the received data to all connected clients
    io.emit('dataUpdate', updatedData);
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected');
  });
});

const port = process.env.SOCKET_PORT || 3001;
httpServer.listen(port, () => {
  console.log(`Socket server listening on port ${port}`);
});