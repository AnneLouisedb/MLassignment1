import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import save_fig, true_false_plot

data = pd.read_csv("datasets/housing.csv")
print(f"shape of data: {data.shape}")
print(data.dtypes)

Xtrain, Xtest, ytrain, ytest = train_test_split(
    data[["longitude", "latitude"]], data["median_house_value"],
)

model = LinearRegression()

model.fit(Xtrain, ytrain)

ypred = model.predict(Xtest)
print("mean absolute error score:", mean_squared_error(ytest, ypred))

true_false_plot(ytest, ypred, "truepred")
