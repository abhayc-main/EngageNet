import React from 'react';
import { Line } from 'react-chartjs-2';

function RealTimeGraph(props) {
    const data = {
        labels: props.timestamps, // Assuming you send timestamps or frame numbers
        datasets: [
            {
                label: 'Number of Interactions per Frame',
                data: props.interactionCounts,
                borderColor: 'blue',
                fill: false,
            },
            {
                label: 'Duration of Interactions',
                data: props.interactionDurations,
                borderColor: 'red',
                fill: false,
            },
        ]
    };

    return (
        <div>
            <Line data={data} />
        </div>
    );
}

export default RealTimeGraph;
