## Algorithm and Data

### Datasets
- **Custom Created Dataset**: Detection of actual top-view heads.
- **New Find**: [Overhead View Dataset](https://universe.roboflow.com/chelkatun-nauka/overhead-view)

    - [People Overhead on Kaggle](https://www.kaggle.com/datasets/hifrom/people-overhead) - 640-1000 files
    - [People Tracking on Kaggle](https://www.kaggle.com/datasets/trainingdatapro/people-tracking?select=images) - 80 files
    - [Medical Staff People Tracking on Kaggle](https://www.kaggle.com/datasets/trainingdatapro/medical-staff-people-tracking) - 70 files

### Angle Dataset
- **Objective**: Get Overhead Crowd Images and Classify them into one of the eight angle classes.
- **Details**: Every Image - an image of cropped head in a full crowd - classified as a certain angle.

### Branches
1. **Proximity Branch**: 
   - Calculate the proximity score as before. This gives us a measure of how close people are to each other.

2. **Angle/Cluster Branch**: 
   - Calculate the engagement score based on the angles of the heads and their spatial grouping. This gives us a measure of how much people are interacting with each other.

3. **Headcount Branch**: 
   - Count the number of heads in the image. This gives us a measure of how many people are present in the scene.

4. **Combine the Scores**: 
   - Aggregate the scores from each branch.

### Main Model
#### Data Collection/Input
- **Head Detection in Real Time Stream**: Trained custom model.

#### Head Cluster Branch:
- Calculate the head orientation score (50% of total score).
- Use a custom CNN model in Tensorflow for angle detection.
- **Data Normalization**: Standardize head center coordinates.
- **Clustering**: Apply DBSCAN clustering.
- **Cluster Metrics Calculation**: Calculate number of clusters and noise points.
- **Engagement Evaluation**: Evaluate group engagement based on orientation.
- **Engagement Score Calculation**: Proportion of "engaged" clusters.
- **Score Capping**: Cap the score at 1 if exceeded.

#### Proximity Analysis Branch:
- Check head center detection.
- Calculate pairwise Euclidean distances between head centers.
- Obtain the median distance.
- Normalize and calculate the proximity score (45% of total score).

#### Headcount Branch:
- Calculate and normalize the headcount score (10% of total score).

#### Combine the Scores:
- Apply weights to each branch's scores.
- Sum or average the scores for overall engagement level.
