// router.mjs

import { Router } from 'next/server';

const api = new Router();

let engagement_score = 0;
let n_clusters = 0;
let n_noise = 0;

api
  .get('/api/data', (req, res) => {
    res.json({
      engagement_score: engagement_score,
      n_clusters: n_clusters,
      n_noise: n_noise
    });
  })
  .post('/api/data', (req, res) => {
    const { engagement_score: es, n_clusters: nc, n_noise: nn } = req.body;
    engagement_score = es;
    n_clusters = nc;
    n_noise = nn;
    res.send('Data updated successfully');
  });

export default api;
