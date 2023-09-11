import { useEffect, useState } from 'react';
import { io, Socket } from 'socket.io-client';

const useSocketData = () => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [data, setData] = useState<{
    engagement_score: number;
    n_clusters: number;
    n_noise: number;
  } | null>(null);

  useEffect(() => {
    const newSocket = io('http://localhost:3001');
    setSocket(newSocket);

    newSocket.on('dataUpdate', (updatedData) => {
      setData(updatedData);
    });

    return () => {
      newSocket.disconnect();
    };
  }, []);

  return data;
};

export default useSocketData;