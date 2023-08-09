### Test cases one by one:

###### Highly Engaged:
- Clusters of people where all are facing the centroid of their cluster.
- Opposite (Highly Unengaged): Clusters of people where all are facing away from the centroid of their cluster


- Engagement Score (Engaged Scenario - Facing Towards Centroid): 1
![engaged](https://github.com/abhayc-glitch/EngageNet/assets/78511893/2a413de2-babd-4896-9312-e955a7c6325a)

- Engagement Score (Unengaged Scenario - Facing Away from Centroid): 0.34310299772711933
![unengaged](https://github.com/abhayc-glitch/EngageNet/assets/78511893/52a39fef-3a4d-48e8-a82e-3a484a9b4022)


Highly Engaged with Noise:
- Similar to the Highly Engaged but with some extra points (people) scattered randomly and facing in random directions

Opposite: 
- Clusters of people where all are facing away from the centroid of their cluster with extra noise points

Engaged with Some Facing Away:
- Clusters where most are facing the centroid, but one person per cluster is facing away

Slightly Unengaged:
- Clusters where most are facing away from the centroid, but one person per cluster is facing towards

Scattered Pairs Facing Each Other:
- Pairs of people scattered throughout the image space, with each pair facing each other

Random Scattering with One Engaged Pair:
- Randomly scattered people with random orientations and one pair facing each other.