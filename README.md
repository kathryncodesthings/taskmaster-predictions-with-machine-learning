# Taskmaster predictions with machine learning

<code> Use machine learning to predict whether early performance can predict long-term outcomes. Your time starts now.</code>

Using historical Taskmaster data, I built a machine learning model to predict contestants’ final series performance using only first-episode results. 

Taskmaster is a  comedy panel game show. In each series of the programme, a group of five celebrities (mainly comedians) attempt to complete a series of challenges, referred to as "tasks". The Taskmaster then reviews the contestants' attempts and awards points based on performance, interpretation or other arbitrary, comedic factors. A winner is determined in each episode and for the series overall. ([Wikipedia](https://en.wikipedia.org/wiki/Taskmaster_(TV_series)))

The project explores how informative early performance is in a partially subjective competition and evaluates whether simple, interpretable models can meaningfully predict long-term outcomes.

## Data sources
The source for this task is a spreadsheet collated by the wonderful Jack Bern ([on Medium](https://jackbern23.medium.com/)), availble in [Google Docs](https://docs.google.com/spreadsheets/d/1S8L34lUyaaV78K02_eAAS-URsKxrWxY1aHT9qKXSoe8/edit?usp=sharing).

## Method
I decided to start with a very simple model, containing:
* Contestant name and series
* Explanatory variables:
  * Points scored per task in Episode 1
  * Score after episode 1
* Scalar response (i.e. what the model will try to predict):
  * % of total points won in series (e.g. in series 13, the Taskmaster awarded 795 points across all tasks to all contestants; the winner of the series got 158 points, or 21.8% of the total points awarded)

Using the % of total points won is useful here because different series have different total points available; this makes all series comparable.

However, this does mean that a contestant’s outcome is not independent of the others in the same series: if one contestant does better, someone else must do worse. This is compositional data. If I let the model treat each contestant separately, the predicted % of points won for a series could be more or less than 100%.

This will be addressed in my model, below. 

And finally, I will show how each contestant was predicted to place given the the forecast % of points won, and compare this to their actual placement. This will be the most interesting, as it's how the show is judged! 

<details> 
  <summary>Taskmaster-specific nerdiness below:</summary>
    Episode 1 could be anomalous due to the prize task and the studio task. This is the first time all the contestants are together in the studio, and many contestants have reflected that they were nervous or under-prepared for the prize/studio tasks at first. So it's possible that some contestants performed worse on these tasks initially before improving. However, because the judging is often highly subjective, it's difficult to isolate this factor from others that affect performance. It might be possible to extend my analysis in the future to review whether there is a group of contestants who perform better/worse on different types of task (studio/pre-recorded/prize tasks) and whether this changes over their series.
</details>

## Notebook 01
The goal of this notebook is to prepare a clean modelling dataset that uses only  information to predict final series performance, and check for nulls or other data errors.

View Notebook 01 here: [Notebook 01](https://github.com/kathryncodesthings/taskmaster-predictions-with-machine-learning/blob/main/notebooks/Notebook%2001.ipynb)

## Notebook 02
Given only  data, how well can we predict final performance?

I used a group-based train/test split so that entire Taskmaster series were held out during testing, ensuring the model was evaluated on completely unseen competitions rather than contestants it had indirectly learned from.

### Baseline model

#### % of points won

I created a baseline model that used an average to forecast contestant performance. This was to provide a comparison to my linear regression model. It produced a mean absolute error (MAE) of 1.5 percentage points, and a root mean squared error (RMSE) of 1.8% points. 

This means that predictions made using a baseline model are off by approx 1.5% percentage points from the actual outcome (using MAE), or using  RMSE (more sensitive to larger errors), 1.8% percentage points.

#### Placement accuracy

However, in practice our model doesn't *need* to accurately predict % of points won. Each contestant finishes the series placing from 1 (most points) to 5 (fewerst points). If it can predict the contestants' final placements in the series, that will be good enough!

Our baseline model forecasts each contestant's placement as follows:

| Series | Mean Absolute Placement Error |
|--------|-------------------------------|
| 1      | 2.0                           |
| 2      | 2.0                           |
| 16     | 2.0                           |
| 18     | 2.0                           |

This gives:
* Mean absolute placement error: 2.0
* Exact placement matches: 4 out of 20

This is compared to the linear regression model, below.

### Linear regression model

The next step was to train the model using linear regression. This tries to estimate a value of y (the scalar response - series overall % of points won) for any value of x (the explanatory variables - performance in Episode 1  and points per task in Episode 1). There are different methods to try to minimise the difference between the linear equation (forecast value) and the actual value (this difference is called the 'residual value').

* The linear regression model vs the baseline model shows a reduced MAE rate from 1.5 to **1.0** percentage points
* The linear regression model vs the baseline model shows a reduced RMSE rate from 1.8 to **1.3** percentage points

This shows that the linear regression model is working better than the simple baseline model.

### Addressing the compositional nature of the data
Final performance is expressed as a percentage of total points awarded within a series, meaning outcomes are compositional and *should* sum to 100% across contestants. The model currently predicts performance independently of this. To respect this contstraint, I rescaled the predicted percentages for each series to total 100.

```py
    y_pred_normalised["Predicted_Final_Pct_Norm"] = (y_pred_normalised.groupby("Series")["Predicted_Final_Pct_Raw"].
    transform(lambda x: x / x.sum()))
```

### Results

#### % of points won

The results of the linear regression model are visualised as follows:

![Scatterplot showing actual % of total points won vs predicted](https://github.com/kathryncodesthings/taskmaster-predictions-with-machine-learning/blob/main/img/Scatterplot%201.png "Scatterplot showing actual % of total points won vs predicted")

An extract of the test data showing how different contestants were expected to perform vs actual performance is shown below, from the highest positive error to lowest negative error:

| Series | Contestant            | Score After Ep 1 | Points Per Task After Ep 1 | Actual % of Total Points Won | Predicted % of Total Points Won | Prediction Error (pp) |
|--------|-----------------------|------------------|----------------------------|------------------------------|---------------------------------|------------------------|
| 18     | Rosie Jones           | 17               | 3.4                        | 18.2%                        | 21.2%                           | 3.0%                   |
| 1      | Roisin Conaty         | 7                | 1.4                        | 15.6%                        | 17.9%                           | 2.3%                   |
| 2      | Joe Wilkinson         | 8                | 1.6                        | 16.6%                        | 18.3%                           | 1.7%                   |
| 16     | Lucy Beaumont         | 13               | 2.6                        | 17.9%                        | 19.2%                           | 1.3%                   |
| 16     | Sam Campbell          | 21               | 4.2                        | 22.3%                        | 21.3%                           | -1.0%                  |
| 2      | Katherine Ryan        | 17               | 3.4                        | 22.5%                        | 20.7%                           | -1.8%                  |
| 1      | Josh Widdicombe       | 13               | 2.6                        | 21.6%                        | 19.5%                           | -2.1%                  |
| 18     | Andy Zaltzman         | 9                | 1.8                        | 21.2%                        | 18.9%                           | -2.3%                  |

Visualising the distribution of errors suggests that it's more common for this model to under-estimate contestants' future % of points won, than to over-estimate them:

![histogram distribution of errors](https://github.com/kathryncodesthings/taskmaster-predictions-with-machine-learning/blob/main/img/Distribution%20of%20errors.png "histogram distribution of errors")

#### Placement accuracy (no spoilers)

| Series | Number of Contestants | Mean Absolute Placement Error |
|--------|----------------------|-------------------------------|
| 1      | 5                    | 1.2                           |
| 2      | 5                    | 0.8                           |
| 16     | 5                    | 0.2                           |
| 18     | 5                    | 1.4                           |

* Mean absolute placement error: 0.90
* Exact placement matches: 11 out of 20

A lower mean absolute error means better ranking accuracy. Series 16 is predicted very accurately overall. Series 18 is the hardest to predict, driven by a large miss at the top of the table.

This supports the idea that early-episode performance is a strong but inconsistent predictor of final ranking.

#### Comparison to baseline model
This is an improvement on the on the baselne model:

| Model             | Mean Absolute Placement Error | Exact placement matches |
|-------------------|-------------------------------|-------------------------|
| Baseline          | 2.0                          | 4 out of 20             |
| Linear regression | 0.9                           | 11 out of 20            |

In the baseline model, which is created using a simple average, all contestants have 20% of the total points end up tied in the same placement. This shows that our model is performing better than simple chance.

#### Next steps

Due to the nature of Taskmaster, contestants can have a low-scoring Episode 1 due to bad luck. It's possible that they will do much better in the remaining episodes. (Or the opposite: a great Episode 1, and scoring low the rest of the series!) So could I improve the model by taking other data into account? There's no accounting for some factors (i.e. the whims of the Taskmaster), but I could try to improve the results.

View Notebook 02 here (main calculations and visualisations): [Notebook 02](https://github.com/kathryncodesthings/taskmaster-predictions-with-machine-learning/blob/main/notebooks/Notebook%2002.ipynb)

View Notebook 04 here (placement calculations): [Notebook 04](https://github.com/kathryncodesthings/taskmaster-predictions-with-machine-learning/blob/main/notebooks/Notebook%2004.ipynb)

## Notebook 03
I compared models trained on Episode 1 data versus Episode 1 **and** 2 data to quantify how much additional predictive 'signal' Episode 2 provides.

![comparing models](https://github.com/kathryncodesthings/taskmaster-predictions-with-machine-learning/blob/main/img/model%201%20vs%20model%202.png "comparison of models")

While additional data is often assumed to improve model performance, incorporating Episode 2 metrics **slightly increased** prediction error. This suggests that early-series volatility, team dynamics, and subjective judging introduce noise that temporarily obscures underlying performance trends. 

#### Placement accuracy (no spoilers)
Placement accuracy also decreases when Episode 2 is incorporated:

| Model             | Mean Absolute Placement Error | Exact placement matches |
|-------------------|-------------------------------|-------------------------|
| Ep 1              | 0.9                           | 11 out of 20            |
| Ep 1 + 2          | 1.05                          |  9 out of 20            |

The result highlights the unpredictable nature of the Taskmaster competition!

View Notebook 03 here: [Notebook 03](https://github.com/kathryncodesthings/taskmaster-predictions-with-machine-learning/blob/main/notebooks/Notebook%2003.ipynb)

## Possible improvements and further exploration
Future model developments could include:
* Repeat this comparison with a tree-based model, which might be more suitable for extracting 'signal' from noisy data
* Turn this into a 'how early can I predict?' curve
* If I exclude team tasks (which introduce complicating factors into individual contestants' performances), does this improve the model accuracy?

This is a very detailed data set which poses some interesting questions:
* Are some contestants provably worse than average at certain kinds of task (prize tasks, team tasks, studio tasks)? Often in the show, a contestant is singled out for being especially bad at, for example, prize tasks or artistic tasks. But is it justified?
* Do certain contestant characteristics statistically affect overall scores? (Age, height, gender, education, number of children, hair colour, etc.)
* Could I combine this with data from [taskmaster.info](https://taskmaster.info/tasks.php)? This site catalogues extremely detailed qualitative information about tasks (e.g. whether a task involves bananas, counting, creativity, hiding things...). It would be interesting to see if we can find further insights from this data as well.
* There are many international Taskmaster versions (Australia, Norway, Canada etc.). Can I work out who's the most harsh Taskmaster judge (based on e.g. disqualifications and point distribution), or if other versions are more or less predictable by our model?
