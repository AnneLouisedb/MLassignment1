import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from src.utils import save_fig, true_false_plot

data = pd.read_csv("datasets/housing.csv")
del data["Unnamed: 0"]

#plots
#scatterplots
data.plot(kind="scatter", x="longitude", y="latitude")
save_fig("graphs/bad_visualization_plot")

data.plot(kind="scatter", x="longitude", y="latitude", alpha = 0.1)
save_fig("graphs/better_visualization_plot")

#income and house value
data.plot(kind="scatter", y="median_income", x="median_house_value", alpha = 0.1)
save_fig("graphs/scatter_income_value")

data.hist(bins=50, figsize=(20,15))
save_fig("graphs/attributes_histogram_plots")

#scattermix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(data[attributes], figsize=(12,8))
save_fig("graphs/scatter_matrix_plot")

#printing shape of data
print(f"shape of data: {data.shape}")
print(data.dtypes)
print(f"description of data: {data.describe()}")

#transforming data
data["income_cat"] = pd.cut(data["median_income"], bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])
data["rooms_per_household"] = data["total_rooms"]/data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"]/data["total_rooms"]
data["population_per_household"]=data["population"]/data["households"]
data = data.dropna()
classifier = LabelEncoder()
print(data.columns)
data["ocean_proximity_numeric"]= classifier.fit_transform(data["ocean_proximity"])
corr_matrix = data.corr()
print(corr_matrix["median_house_value"].apply(abs).sort_values(ascending=False))

#rooms per household is more correlated with the house value than total_rooms

data.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])


# Training and Testing models
Xtrain, Xtest, ytrain, ytest = train_test_split(
    data[["median_income",
          "bedrooms_per_room",
          "rooms_per_household",
          "housing_median_age",
          "ocean_proximity_numeric",
          "latitude"]], data["median_house_value"], random_state=30)

model = LinearRegression()

model.fit(Xtrain, ytrain)

ypred = model.predict(Xtest)
print("mean absolute error score:", mean_squared_error(ytest, ypred))

model2 = RandomForestRegressor()
model2.fit(Xtrain,ytrain)
ypred2 = model2.predict(Xtest)
print("random forest: mean absolute error score:", mean_squared_error(ytest, ypred2))
true_false_plot(ytest, ypred, "truepred1")
plt.show()


