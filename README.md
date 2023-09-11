![Screenshot 2023-09-10 at 8 49 58 PM](https://github.com/abhayc-glitch/EngageNet/assets/78511893/60ad3ee4-78ce-409a-bd7a-7e96dd917f0d)

##### A crowd/audience engagement measurement system using multiple metrics.

### Metrics/Code flow
MAIN MODEL

Processing Method - Real-time Frame Detection - Allows for more insights on crowd - processing power will be handled with Intel Neural Processing Stick

1. Data Collection:
    Gather a labeled dataset of crowd videos captured from a top-down camera perspective.
    Annotate the videos with ground truth labels indicating the engagement level or health of interactions within the crowd.
    Get around a 200 data images - train model
    - Use this data for both Head Angle Calculation and Head Detection

##### Metrics for Detection (Summary)
- Proximity branch: We calculate the proximity score as before. This gives us a measure of how close people are to each other.
- Angle/cluster branch: We calculate the engagement score based on the angles of the heads and their spatial grouping. This gives us a measure of how much people are interacting with each other.
- Headcount branch: We count the number of heads in the image. This gives us a measure of how many people are present in the scene.
Combine the scores for these


##### Metrics for Detection - Code
- Head Orientation - 50%
    Count the total number of heads in the group (N).
    Count the number of heads in the group facing towards the rest of the group (F).
    Calculate the score by dividing F by N and multiplying
- Proximity Analysis - 40%
    Calculate the spatial distance between annotated heads. - Distance Matrix
    Identify clusters of heads based on their proximity to each other.
    Assign a higher proximity score to heads that belong to larger clusters, indicating higher engagement.
    Normalize the proximity scores between 0 and 1, where 0 represents minimal proximity and 1 represents maximum proximity engage
- Headcount Branch - 5% of total score


###### Angle Classes
| Angle | Direction |
|-------|-----------|
| 0     | Down      |
| 45    | Down-Right|
| 90    | Right     |
| 135   | Up-Right  |
| 180   | Up        |
| 225   | Up-Left   |
| 270   | Left      |
| 315   | Down-Left |


###### Angle Data
0: 2870 files
135: 2986 files
180: 2832 files
225: 2932 files
270: 2800 files
315: 2978 files
45: 2996 files
90: 2806 files


###### Demo
![Screenshot 2023-09-10 at 8 27 50 PM](https://github.com/abhayc-glitch/EngageNet/assets/78511893/8b7a01a3-9fb6-4e21-90f8-29790208eb2c)

