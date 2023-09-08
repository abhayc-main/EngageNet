"use client"

import React, { useEffect, useState } from 'react';
import Layout from './layout';
import Card from '/Users/abhay/Documents/EngageNet/client/components/home/card';
import { LineChart, Line } from 'recharts';
import io from 'socket.io-client';

type DataType = {
  score: number;
  // add other properties if there are any
};

export default function Page() {
  const [data, setData] = useState<DataType[]>([]);

  useEffect(() => {
    // Connect to the ./api/data endpoint
    const socket = io('/api/data');

    socket.on('updateData', (newData) => {
      setData(newData); // Update data whenever the server emits 'updateData'
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  return (
    <div>
      <h1>Hello</h1>
      <div className="w-full max-w-4xl space-y-6">
        <Card>
          <h2 className="text-xl font-bold mb-4">Engagement Scores Graph</h2>
          <LineChart width={400} height={200} data={data}>
            <Line type="monotone" dataKey="score" stroke="#8884d8" />
          </LineChart>
        </Card>
        <Card>
          <h2 className="text-xl font-bold mb-4">Latest Engagement Score</h2>
          <p className="text-lg">
            {data.length > 0 ? data[data.length - 1].score : 'Loading...'}
          </p>
        </Card>
      </div>
    </div>
  );
}
