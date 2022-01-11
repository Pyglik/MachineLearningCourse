import DARTS
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree
from DARTS.models.forecasting.regression_model import RegressionModel
from DARTS.models.forecasting import NaiveDrift, NaiveSeasonal, ExponentialSmoothing, Theta
from DARTS.metrics import mape
from DARTS.utils.statistics import check_seasonality

df = pd.read_csv('AirPassengers.csv')
df.plot()
plt.show()

series = DARTS.TimeSeries.from_dataframe(df, 'Month', '#Passengers')
print(series)
series.plot()
plt.show()

train, test = series[:-36], series[-36:]
train.plot()
test.plot()
plt.show()

model = NaiveDrift()
model.fit(train)
predicted = model.predict(len(test))
print(predicted)
predicted.plot()
test.plot()
plt.show()
score = mape(actual_series=test, pred_series=predicted)
print(score)

for m in range(2, len(train)):
    is_seasonal, period = check_seasonality(train, m=m, max_lag=len(train))
    if is_seasonal:
        print('I seasonal! ', period)

model = NaiveSeasonal(K=12)
model.fit(train)
predicted2 = model.predict(len(test))
print(predicted2)
predicted2.plot()
test.plot()
plt.show()
score = mape(actual_series=test, pred_series=predicted2)
print(score)

predicted_sum = predicted + predicted2 - train.last_value()
print(predicted_sum)
predicted_sum.plot()
test.plot()
plt.show()
score = mape(actual_series=test, pred_series=predicted_sum)
print(score)

model = ExponentialSmoothing()
model.fit(train)
predicted2 = model.predict(len(test))
print(predicted2)
predicted2.plot()
test.plot()
plt.show()
score = mape(actual_series=test, pred_series=predicted2)
print(score)

model = Theta()
model.fit(train)
predicted2 = model.predict(len(test))
print(predicted2)
predicted2.plot()
test.plot()
plt.show()
score = mape(actual_series=test, pred_series=predicted2)
print(score)

model = RegressionModel(sklearn.tree.DecisionTreeRegressor)
model.fit(train)
predicted2 = model.predict(len(test))
print(predicted2)
predicted2.plot()
test.plot()
plt.show()
score = mape(actual_series=test, pred_series=predicted2)
print(score)
