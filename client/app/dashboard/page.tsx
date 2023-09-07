// page.js

import React, { useEffect, useState } from 'react';
import Layout from './layout';
import Card from './components/card';
import { LineChart, Line } from 'recharts';
import io from 'socket.io-client';

export default function Page() {
  const [data, setData] = useState([]);

  useEffect(() => {
    const socket = io('/'); // Connect to the server

    socket.on('updateData', (newData) => {
      setData(newData); // Update data whenever the server emits 'updateData'
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  return (
    <Layout>
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
    </Layout>
  );
}
