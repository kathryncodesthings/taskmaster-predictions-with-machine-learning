**Watching Taskmaster Series 21**

For the first time, I'm going to use this model live, and see if it can predict the winner.

My model is relatively basic in forecasting placement - the person with the highest points in Episode 1 is usually forecast to win the series. 

However this is quite a successful model, as it results in a mean absolute placement error of **0.90**. 

Having used the training data on Episode 1 of Series 21, the model forecasts that the contestants will perform as follows:

| Series | Contestant         | score_ep1 | ppt_ep1 | Predicted_Final_Pct_Norm  | forecast_placement |
|--------|--------------------|-----------|---------|---------------------------|--------------------|
| 21     | Amy Gledhill       | 16        | 3.2     | 0.206002                  | 2                  |
| 21     | Armando Iannucci   | 11        | 2.2     | 0.192362                  | 3                  |
| 21     | Joanna Page        | 22        | 4.4     | 0.222369                  | 1                  |
| 21     | Joel Dommett       | 11        | 2.2     | 0.192362                  | 3                  |
| 21     | Kumail Nanjiani    | 9         | 1.8     | 0.186906                  | 5                  |


View Series 21 - Forecast results notebook here: [Series 21 - Forecast results](https://github.com/kathryncodesthings/taskmaster-predictions-with-machine-learning/blob/main/notebooks/Series%2021%20-%20Forecast%20results.ipynb)
