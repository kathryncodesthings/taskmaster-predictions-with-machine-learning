# Taskmaster predictions with machine learning

<code> Use machine learning to predict whether early performance can predict long-term outcomes. Your time starts now.</code>

Taskmaster is a  comedy panel game show. In each series of the programme, a group of five celebrities (mainly comedians) attempt to complete a series of challenges, referred to as "tasks". The Taskmaster then reviews the contestants' attempts and awards points based on performance, interpretation or other arbitrary, comedic factors. A winner is determined in each episode and for the series overall. ([Wikipedia](https://en.wikipedia.org/wiki/Taskmaster_(TV_series)))

Using historical Taskmaster data, I built a machine learning model to predict contestants’ final series performance using only first-episode results. 

The project explores how informative early performance is in a partially subjective competition and evaluates whether simple, interpretable models can meaningfully predict long-term outcomes.

## Data sources
The source for this task is a spreadsheet carefully collated by the wonderful Jack Bern ([on Medium](https://jackbern23.medium.com/)), availble in [Google Docs](https://docs.google.com/spreadsheets/d/1S8L34lUyaaV78K02_eAAS-URsKxrWxY1aHT9qKXSoe8/edit?usp=sharing).

## Notebook 01
The goal of this notebook is to prepare a clean modelling dataset that uses only Episode 1 information to predict final series performance, and check for nulls or other data errors.

View Notebook 01 here: [Notebook 01](https://github.com/kathryncodesthings/taskmaster-predictions-with-machine-learning/blob/main/notebooks/Notebook%2001.ipynb)

## Notebook 02
Given only Episode 1 data, how well can we predict final performance?

I used a group-based train/test split so that entire Taskmaster series were held out during testing, ensuring the model was evaluated on completely unseen competitions rather than contestants it had indirectly learned from.

I then created a baseline model that used an average to forecast contestant performance. This was to provide a comparison to my linear regression model. It produced a mean absolute error (MAE) of 1.5 percentage points, and a root mean squared error (RMSE) of 1.8% points. 
This means that predictions made using a baseline model are off by approx 1.5% percentage points from the actual outcome (using MAE), or using  RMSE (more sensitive to larger errors), 1.8% percentage points.

The next step was to train the model using linear regression. This tries to estimate a value of y (the scalar response - series overall performance) for any value of x (the explanatory variables - performance in episode 1 and points per task in episode 1). There are different methods to try to minimise the difference between the linear equation (forecast value) and the actual value (this difference is called the 'residual value').

* The linear regression model vs the baseline model shows a reduced MAE rate from 1.5 to **1.1** percentage points
* The linear regression model vs the baseline model shows a reduced RMSE rate from 1.8 to **1.4** percentage points

This shows that the linear regression model is working better than the simple baseline model.

The results of the linear regression model are visualised as follows:

![Scatterplot showing actual % of total points won vs predicted](https://github.com/kathryncodesthings/taskmaster-predictions-with-machine-learning/blob/main/img/Scatterplot%201.png "Scatterplot showing actual % of total points won vs predicted")

An extract of the test data showing how different contestants were expected to perform vs actual performance is shown below, from the highest positive error to lowest negative error:

| Series | Contestant            | Score After Ep 1 | Points Per Task Score After Ep 1 | Actual % of Total Points Won in Series | Predicted % of Total Points Won in Series | Prediction Error (Percentage Points) |
|--------|----------------------|----------------|---------------------------------|-------------------------------------------------|---------------------------------------------------|-------------------------------------|
| 18     | Rosie Jones           | 17             | 3.4                             | 18.2%                                           | 20.7%                                             | 2.5%                                |
| 1      | Roisin Conaty         | 7              | 1.4                             | 15.6%                                           | 18.0%                                             | 2.4%                                |
| 16     | Lucy Beaumont         | 13             | 2.6                             | 17.9%                                           | 19.6%                                             | 1.7%                                |
| 2      | Joe Wilkinson         | 8              | 1.6                             | 16.6%                                           | 18.2%                                             | 1.7%                                |
| 2      | Richard Osman         | 20             | 4.0                             | 20.6%                                           | 21.5%                                             | 0.9%                                |
| 18     | Babatunde Aléshé      | 9              | 1.8                             | 19.5%                                           | 18.5%                                             | -1.0%                               |
| 2      | Katherine Ryan        | 17             | 3.4                             | 22.5%                                           | 20.7%                                             | -1.9%                               |
| 1      | Josh Widdicombe       | 13             | 2.6                             | 21.6%                                           | 19.6%                                             | -2.0%                               |
| 18     | Andy Zaltzman         | 9              | 1.8                             | 21.2%                                           | 18.5%                                             | -2.7%                               |

A different way to visualise the data is in using a dumb-bell chart, which makes it easier to spot which contestants had the most variation between predicted % of points scored and actual:

![Dumb-bell chart of errors](https://github.com/kathryncodesthings/taskmaster-predictions-with-machine-learning/blob/main/img/dumbell%20chart.png "Dumb-bell chart of errors")

<details> 
  <summary>Brief discussion of results with spoilers for Taskmaster series 1, 2 and 18</summary>
   It's interesting that the winners of series 1, 2, and 18 were the **most** underscored by the model. 
</details>

Visualising the distribution of errors suggests that it's more common for this model to under-estimate contestants' future % of points won, than to over-estimate them:

![histogram distribution of errors](https://github.com/kathryncodesthings/taskmaster-predictions-with-machine-learning/blob/main/img/Distribution%20of%20errors.png "histogram distribution of errors")


Due to the nature of Taskmaster, contestants can have a low-scoring episode 1 due to bad luck. It's possible that they will do much better in the remaining episodes. (Or the opposite: a great episode 1, and scoring low the rest of the series!) So could I improve the model by taking other data into account? There's no accounting for some factors (e.g. the Taskmaster's comedic vendetta against certain contestants), but I could try to improve the results.

View Notebook 02 here: [Notebook 02](https://github.com/kathryncodesthings/taskmaster-predictions-with-machine-learning/blob/main/notebooks/Notebook%2002.ipynb)

## Notebook 03
I compared models trained on Episode 1 data versus Episode 1–2 data to quantify how much additional predictive 'signal' Episode 2 provides.

![comparing models](https://github.com/kathryncodesthings/taskmaster-predictions-with-machine-learning/blob/main/img/model%201%20vs%20model%202.png "comparison of models")

While additional data is often assumed to improve model performance, incorporating Episode 2 metrics **slightly increased** prediction error. This suggests that early-series volatility, team dynamics, and subjective judging introduce noise that temporarily obscures underlying performance trends. 

The result highlights the unpredictable nature of the Taskmaster competition!

View Notebook 03 here: [Notebook 03](https://github.com/kathryncodesthings/taskmaster-predictions-with-machine-learning/blob/main/notebooks/Notebook%2003.ipynb)

## Possible improvements and further exploration
Future model developments could include:
* Repeat this comparison with a tree-based model, which might be more suitable for extracting 'signal' from noisy data
* Turn this into a 'how early can we predict?' curve

This is a very detailed data set which poses some interesting questions:
* Are some contestants provably worse than average at certain kinds of task (prize tasks, team tasks, studio tasks)?
* If we exclude team tasks (which introduce futher complicating factors into individual contestants' performances), does this improve the model accuracy?
* Do certain contestant characteristics statistically affect overall scores? (Age, height, gender, education, number of children, hair colour, etc.)
* Could I combine this with data from [taskmaster.info](https://taskmaster.info/tasks.php)? This site catalogues extremely detailed qualitative information about tasks (e.g. whether a task involves bananas, counting, creativity, hiding things...). It would be interesting to see if we can find further insights from this data as well.
