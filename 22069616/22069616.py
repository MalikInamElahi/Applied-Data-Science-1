# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 23:47:07 2024

@author: Malik InamElahi
"""
# All Libraries Add 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

# File paths for the datasets 
data = 'data1/API_SP.POP.TOTL_DS2_en_csv_v2_6298256.csv'
country_file = 'data1/Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_6298256.csv'
indicator_file = 'data1/Metadata_Indicator_API_SP.POP.TOTL_DS2_en_csv_v2_6298256.csv'

### data files read function 
def read_data(file_paths):
    return pd.read_csv(file_paths[0], skiprows=4), pd.read_csv(file_paths[1]), pd.read_csv(file_paths[2])

# data clean 
def clean_data(df):
    df = df.dropna(axis=1, how='all')
    year_columns = df.columns[4:]
    df[year_columns] = df[year_columns].fillna(0)
    df = df.drop_duplicates()
    df[year_columns] = df[year_columns].astype(float)
    for year in year_columns:
        mean_value = df[year].mean()
        std_dev = df[year].std()
        outlier_threshold = mean_value + 3 * std_dev
        df[year] = df[year].clip(upper=outlier_threshold)

    return df
# cluster function 
def plot_clusters(df, centers):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='2000', y='2020', hue='Cluster', legend='full')

# For loop for data itrate 
    for i, center in enumerate(centers):
        plt.scatter(center[0], center[1], color='black', marker='X', s=100, edgecolor='w', label=f'Center {i+1}' if i == 0 else None)

    plt.title('Clusters of Total Population Change (2000 vs 2020)')
    plt.xlabel('Total Population in 2000 (%)')
    plt.ylabel('Total Population in 2020 (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

## Function data histrogram
def plot_histogram(data):
    plt.figure(figsize=(8, 6))
    sns.histplot(data['2020'].dropna(), kde=True)
    plt.title('Distribution of Total Population in 2020')
    plt.xlabel('Total Population (%)')
    plt.ylabel('Frequency')
    plt.show()

## function for box plot

def plot_boxplot(data, subset):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data[data['Country Name'].isin(subset)], x='Country Name', y='2020')
    plt.title('Boxplot of Total Population in 2020 by Country')
    plt.xlabel('Country')
    plt.ylabel('Total Population (%)')
    plt.xticks(rotation=45)
    plt.show()

def plot_scatter_comparison(data, x_column, y_column, color='red'):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=x_column, y=y_column, color=color)
    plt.title(f'Comparative Scatter Plot of Total Population ({x_column} vs {y_column})')
    plt.xlabel(f'Total Population in {x_column} (%)')
    plt.ylabel(f'Total Population in {y_column} (%)')
    plt.show()

def cluster_and_plot(data, scaler, num_clusters):
    cluster_df = data[['Country Name', '2000', '2020']].dropna()
    cluster_df['Change_2000_2020'] = cluster_df['2020'] - cluster_df['2000']

    normalized_data = scaler.fit_transform(cluster_df[['2000', '2020', 'Change_2000_2020']])

    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_df['Cluster'] = kmeans.fit_predict(normalized_data)

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    plot_clusters(cluster_df, centers)

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def calculate_fit_and_predict(data, model_func, years_to_predict):
    x_axis_data = np.array(range(len(data)))
    y_axis_data = data.values
    params, _ = curve_fit(model_func, x_axis_data, y_axis_data, maxfev=10000)

    # Predict future values
    future_x = np.array(range(len(data) + years_to_predict))
    future_y = model_func(future_x, *params)

    return future_x, future_y, params

def calculate_error_ranges(params, cov, x):
    partials = np.array([x**i for i in range(len(params))])
    sigma_y = np.sqrt(np.sum((cov @ partials)**2, axis=0))
    return sigma_y

# Read data
file_paths = [data, country_file, indicator_file]
orig_population_data, country, indicator = read_data(file_paths)

# Clean data
population_data = clean_data(orig_population_data)

# Data for clustering
scaler = StandardScaler()
cluster_and_plot(population_data, scaler, 5)

# Time series plot
sns.set_style("whitegrid")
sample_countries = ['United States', 'China', 'India', 'Germany', 'Pakistan', 'Afghanistan']
time_series_data = population_data[population_data['Country Name'].isin(sample_countries)]

# Selecting the last 33 years
last_33_years = time_series_data.columns[-33:]

plt.figure(figsize=(12, 6))
for country in sample_countries:
    plt.plot(last_33_years, time_series_data[time_series_data['Country Name'] == country][last_33_years].iloc[0], label=country)

plt.title('Trend of Total Population (1990-2022)')
plt.xlabel('Year')
plt.ylabel('Total Population (%)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Histogram
plot_histogram(population_data)

# Boxplot
region_subset = ['United States', 'China', 'India', 'Germany', 'Afghanistan', 'Australia', 'Nigeria', 'Russia', 'Pakistan', 'South Africa']
plot_boxplot(population_data, region_subset)

# Scatter plot
plot_scatter_comparison(population_data, '2010', '2020', color='red')

cluster_years = ['1990', '2000' ,'2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
cluster_data = population_data.set_index('Country Name')[cluster_years].dropna()

# Applying K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=0).fit(cluster_data)
cluster_data['Cluster'] = kmeans.labels_

# Visualizing the Clusters
plt.figure(figsize=(12, 6))
for cluster in np.unique(kmeans.labels_):
    cluster_subset = cluster_data[cluster_data['Cluster'] == cluster]
    plt.scatter(cluster_subset['1990'], cluster_subset['2020'], label=f'Cluster {cluster}')

plt.title('Cluster Plot of Total Population (1990 vs 2020)')
plt.xlabel('Total Population in 1990 (%)')
plt.ylabel('Total Population in 2020 (%)')
plt.legend()
plt.show()

# Selecting one country from each cluster for the analysis
selected_countries = cluster_data.reset_index().groupby('Cluster')['Country Name'].last()
selected_countries_list = selected_countries.tolist()

# Preparing data for curve fitting
fit_years = [str(year) for year in range(2000, 2021)]
fit_data = population_data[population_data['Country Name'].isin(selected_countries_list)]
fit_data = fit_data[['Country Name'] + fit_years].set_index('Country Name')

years_to_predict = 20
# Fitting models and predicting for each selected country
predictions = {}
for country in selected_countries_list:
    data = fit_data.loc[country]
    future_x, future_y, _ = calculate_fit_and_predict(data, exponential, years_to_predict)
    predictions[country] = (future_x, future_y)

# Displaying the predictions
predictions[selected_countries_list[0]]

# Visualizing the best fitting function for each country
plt.figure(figsize=(15, 10))
for i, country in enumerate(selected_countries_list):
    data = fit_data.loc[country]
    x_axis_data = np.array(range(len(data)))
    y_axis_data = data.values

    # Model Curve
    params, cov = curve_fit(exponential, x_axis_data, y_axis_data, maxfev=10000)
    future_x = np.array(range(len(data) + years_to_predict))
    future_y = exponential(future_x, *params)

    # Calculate Error ranges
    sigma_y = calculate_error_ranges(params, cov, future_x)

    # Plotting
    plt.subplot(3, 2, i+1)
    plt.plot(future_x[:len(data)], y_axis_data, 'o', label=f'Actual Data ({country})')
    plt.plot(future_x, future_y, '-', label='Fitted Model')
    plt.fill_between(future_x, future_y - sigma_y, future_y + sigma_y, alpha=0.2)
    plt.title(country)
    plt.xlabel('Years since 2000')
    plt.ylabel('Total Population (%)')
    plt.legend()

plt.tight_layout()
plt.show()

##  The provided Python code performs data analysis and clustering on global population data, 
## focusing on the years 2000 and 2020. It uses various data visualization techniques, including scatter plots, 
## histograms, and box plots, to analyze population trends. The K-Means clustering algorithm is applied to 
## identify differrent clusters of countries based on their population changes. Additionally, the code fits an 
## exponential growth model to predict future population trends for selected countries, 
## considering the years 2000 to 2020. The results are visualized with fitted curves and error ranges. 
## The code is well-organized, leveraging popular libraries like pandas, seaborn, and scikit-learn. 

## Github Link ##
### https://github.com/MalikInamElahi/Applied-Data-Science-1.git