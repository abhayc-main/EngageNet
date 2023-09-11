// ./pages/api/data.ts
import type { NextApiRequest, NextApiResponse } from 'next'

type Data = {
  engagement_score: number,
  n_clusters: number,
  n_noise: number
}

export default function handler(req: NextApiRequest, res: NextApiResponse<Data>) {
  if (req.method === 'POST') {
    const data: Data = req.body
    res.status(200).json(data)
  } else {
    res.status(405).end() // Method Not Allowed
  }
}
