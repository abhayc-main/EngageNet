// pages/api/data.js

let engagement_score = 0;
let n_clusters = 0;
let n_noise = 0;

export default (req : any, res : any) => {
  if (req.method === 'POST') {
    engagement_score = req.body.engagement_score;
    n_clusters = req.body.n_clusters;
    n_noise = req.body.n_noise;
    res.status(200).send('Data updated successfully');
  } else {
    res.status(200).json({
      engagement_score: engagement_score,
      n_clusters: n_clusters,
      n_noise: n_noise
    });
  }
};
