import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.utils import save_fig, true_false_plot

data = pd.read_csv("datasets/housing.csv")
#printing shape of data
print(f"shape of data: {data.shape}")
print(data.dtypes)
print(f"description of data: {data.describe()}")



Xtrain, Xtest, ytrain, ytest = train_test_split(
    data[["longitude", "latitude"]], data["median_house_value"],
)

model = LinearRegression()

model.fit(Xtrain, ytrain)

ypred = model.predict(Xtest)
print("mean absolute error score:", mean_squared_error(ytest, ypred))

true_false_plot(ytest, ypred, "truepred")

#scatterplots
data.plot(kind="scatter", x="longitude", y="latitude")
save_fig("bad_visualization_plot")

data.plot(kind="scatter", x="longitude", y="latitude", alpha = 0.1)
save_fig("better_visualization_plot")

#scattermix
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(data[attributes], figsize=(12,8))
save_fig("scatter_matrix_plot")

#income and house value
data.plot(kind="scatter", y="median_income", x="median_house_value", alpha = 0.1)
save_fig("scatter_income_value")

data.hist(bins=50, figsize=(20,15))
save_fig("attributes_histogram_plots")

#second linear regression
Xtrain2, Xtest2, ytrain2, ytest2 = train_test_split(
    data[["median_income"]], data["median_house_value"],
 test_size=0.2, random_state=42)
print(f"test set head:{Xtest2.head()}")
model.fit(Xtrain2, ytrain2)

ypred2 = model.predict(Xtest2)
print("mean absolute error score2:", mean_squared_error(ytest2, ypred2))

true_false_plot(ytest2, ypred2, "truepred2")



data["income_cat"] = pd.cut(data["median_income"], bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])
print(data["income_cat"].value_counts())
data["income_cat"].hist()



#stratify sampling based on the income category

#from sklearn.model_selection import StratifiedShuffleSplit



data["rooms_per_household"] = data["total_rooms"]/data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"]/data["total_rooms"]
data["population_per_household"]=data["population"]/data["households"]

corr_matrix = data.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
#rooms per household is more correlated with the house value than total_rooms

data.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])

#prepare data for linear regression


