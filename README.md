![Screenshot 2023-09-10 at 8 49 58 PM](https://github.com/abhayc-glitch/EngageNet/assets/78511893/60ad3ee4-78ce-409a-bd7a-7e96dd917f0d)

### A crowd/audience engagement measurement leveraging Overhead Cameras with Clustering and Euclidean Metrics

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

