<h1 align="center">EngageNet</h1>
<p align="center">
  The nextjs app to go along with the main project code
</p>


#### Enhancing Scientific Interactions in Conferences: A Novel Approach Using Overhead Camera Pose Estimation with Clustering and Euclidean Metrics

#### Features
**Overhead Camera Pose Estimation:** Utilizing overhead camera pose estimation, the system effectively tracks and analyzes the dynamics of scientific discussions, providing unparalleled insight into interaction patterns.

**Live Interaction Analysis:** Cameras strategically placed over the conference venue capture live footage of researchers during pivotal moments of their discussions, offering a real-time view into the heart of the conference.

**Data-Driven Conference Improvement:** Beyond merely capturing interactions, this project enables organizers to use the acquired data to make informed decisions, optimize conference layouts, and enhance the overall attendee experience.

**Advanced Scoring using euclidean metrics:** By assessing parameters such as proximity, duration, and frequency of interactions, it calculates a comprehensive 'interaction score' that offers quantifiable insights into the effectiveness of exchanges among researchers. 

## Get Started

### Installation

#### Models
1. Acquire the trained models (.pt yolo files) -> email me at abhichebium@gmail.com for access
2. Put the downloaded models in the `./engagenet/models/` folder
3. Make sure the names are `best.pt` and `angle_best.pt`

#### UI
1. Clone the whole repository and navigate to the `./client` folder
2. Make sure you have `Node.js` and `npm/yarn` installed.
3. Run `npm install` in the `./client` root directory to install all the dependecies

#### Detection/Algorithm code
1. Navigate to the `./engagenet` folder
2. Make sure to have `python 3.9.6` installed.
3. Activate a virtual env if you prefer it
4. Run `pip install -r requirements.txt`

### Usage
###### At this point your folder should look like this
<img width="126" alt="image" src="https://github.com/abhayc-glitch/EngageNet/assets/78511893/3796a406-edd9-47bd-9a43-cc76e11aa714">


1. I recommend having 3 different terminal windows - client, socket server and python detection code
2. In the first window, navigate to the `./client` directory and run `npm run dev` to start the `NextJS` app
3. In the second window and the same directory run `npm run start:socket`, this will start the socketIO server
4. Now navigate to the `./engagenet` directory and run `python main.py`
> ⚠️: **Make sure to connect your Webcam/Camera through USB**: Switch up the camera source in `main.py` if your camera is not being recognized
5. Now drag the python window that pops with the live feed on top of the UI box that says place here
> In the end it should look like this
> 
![Screenshot 2023-09-10 at 8 27 50 PM](https://github.com/abhayc-glitch/EngageNet/assets/78511893/8b7a01a3-9fb6-4e21-90f8-29790208eb2c)


### CLI
If you want to use just the CLI and not the UI, navigate to the `./engagenet` directory and run `python cli.py`
> It should end up looking like this
>
<img width="796" alt="image" src="https://github.com/abhayc-glitch/EngageNet/assets/78511893/7af4c46b-4ac2-45aa-9b9a-383e48690d90">


## Code Flow

### Real-time Frame Detection System

#### 1. Data Collection:
- **Objective:** Gather and annotate a dataset for training a model to assess crowd engagement and interaction health.
- **Method:**
  - Collect a dataset of approximately 200 crowd videos from a top-down camera perspective.
  - Annotate videos with ground truth labels indicating levels of engagement or interaction health.
  - Utilize this dataset for both Head Angle Calculation and Head Detection.

#### 2. Metrics for Detection (Summary):
- **Proximity Branch:** 
  - Calculate a proximity score indicating how close individuals are to each other.
- **Angle/Cluster Branch:** 
  - Determine engagement score based on head angles and spatial clustering.
- **Headcount Branch:** 
  - Count the number of heads in the image to assess crowd size.
- **Overall Scoring:** 
  - Combine scores from the above branches for a comprehensive engagement analysis.

#### 3. Metrics for Detection - Detailed Approach:
- **Head Orientation (50% of total score):** 
  - Count total heads (N) and heads facing towards the group (F).
  - Score = (F / N) * 100.
- **Proximity Analysis (40% of total score):** 
  - Calculate distances between heads using a distance matrix.
  - Identify and score head clusters based on proximity.
  - Normalize scores between 0 (minimal proximity) to 1 (maximum proximity).
- **Headcount Branch (5% of total score):** 
  - Simply count the number of heads in the scene.

#### 4. Angle Classes:
- Representation of head orientation angles and their corresponding directions:

| Angle | Direction  |
| ----- | ---------- |
| 0     | Down       |
| 45    | Down-Right |
| 90    | Right      |
| 135   | Up-Right   |
| 180   | Up         |
| 225   | Up-Left    |
| 270   | Left       |
| 315   | Down-Left  |

#### 5. Angle Data:
- Distribution of files across different angle classes:

| Angle | Number of Files |
| ----- | --------------- |
| 0     | 2870            |
| 45    | 2996            |
| 90    | 2806            |
| 135   | 2986            |
| 180   | 2832            |
| 225   | 2932            |
| 270   | 2800            |
| 315   | 2978            |

