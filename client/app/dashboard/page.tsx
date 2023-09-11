// dashboard/page.tsx

"use client"

import useSocketData from '/Users/abhay/Documents/EngageNet/client/lib/hooks/useSocketData';
import { Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle, } from "@/components/ui/card";

import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis } from "recharts"
import { LineChart, Line, CartesianGrid, Tooltip } from 'recharts';


import { useEffect, useState } from 'react';

import { io } from 'socket.io-client';

const socket = io('http://localhost:3001');


export default function DashboardPage() {

  const [data, setData] = useState({
    engagement_score: 0,
    n_total: 0,
    n_clusters: 0,
    n_noise: 0,
  });

  useEffect(() => {
    socket.on('dataUpdate', (newData) => {
      setData(newData);
    });

    return () => {
      socket.off('dataUpdate');
    };
  }, [])

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Engagement Score
          </CardTitle>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            className="h-4 w-4 text-muted-foreground"
          >
            <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
          </svg>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{data.engagement_score}</div>
          <p className="text-xs text-muted-foreground">
            Score out of 1.0
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Total Number of People
          </CardTitle>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            className="h-4 w-4 text-muted-foreground"
          >
            <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
            <circle cx="9" cy="7" r="4" />
            <path d="M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
          </svg>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{data.n_total}</div>
          <p className="text-xs text-muted-foreground">
            People present in the area
          </p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Number of Clusters</CardTitle>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            className="h-4 w-4 text-muted-foreground"
          >
            <rect width="20" height="14" x="2" y="5" rx="2" />
            <path d="M2 10h20" />
          </svg>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{data.n_clusters}</div>
          <p className="text-xs text-muted-foreground">
            Clusters Detected
          </p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Noise
          </CardTitle>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            className="h-4 w-4 text-muted-foreground"
          >
            <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
          </svg>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{data.n_noise}</div>
          <p className="text-xs text-muted-foreground">
            Noise Subjects Detected
          </p>
        </CardContent>
      </Card>

      <Card className="col-span-5 row-span-5">
        <CardHeader>
          <CardTitle>Live Feed</CardTitle>
          <CardDescription>
            Put your Python Window in this box
          </CardDescription>
        </CardHeader>
        <CardContent className="p-40 h-screen"> {/* Adjust padding and height here */}
          {/* Place for the Python window */}
        </CardContent>
      </Card>


      
    </div>
    
  );
}