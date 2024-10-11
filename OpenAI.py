# Creating a dummy dataset

import pandas as pd
import numpy as np
from random import choices
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

# Fixing the random seed for reproducibility
np.random.seed(10)

# Parameters for the dummy dataset
num_record = 1000
num_outlet = 10
num_products = 20

# Creating a date range for two quarters (Jan-Mar, Apr-Jun)
dates = pd.date_range(start='1/1/2023', end='30/06/2023')
dates = choices(dates, k=num_record)

# Creating outlet names
outlets = ["outlet_" + str(i + 1) for i in range(num_outlet)]
outlets = choices(outlets, k=num_record)

# Creating product names
products = ["product_" + str(i + 1) for i in range(num_products)]
products = choices(products, k=num_record)

# Generating random data for sales
units_sold = np.random.randint(1, 200, num_record)
price_per_unit = np.random.uniform(10, 500, num_record)
total_sales = units_sold * price_per_unit

# Creating the DataFrame
df = pd.DataFrame({
    "Date": dates,
    "Outlet": outlets,
    "Product": products,
    "Units Sold": units_sold,
    "Price per unit": price_per_unit,
    "Total Sales": total_sales
})

# Initialize OpenAI with your API key
# IMPORTANT: Avoid sharing your API key in public code
llm = OpenAI(api_token="your api key")

# Initialize PandasAI
pandas_ai = PandasAI(llm)

# Run the analysis using PandasAI to find the product with the highest total sales
highest_product = pandas_ai.run(df , "Which product has the highest total sales?")
print(highest_product)

pandas_ai.run(df, "Format Price_per_Unit and Total_Sales columns to 2 decimal places")
pandas_ai.run(df, "Show the first 5 rows of the dataset")
pandas_ai.run(df, "Show the last 5 rows of the dataset")
pandas_ai.run(df, "Show the shape of the dataset")
pandas_ai.run(df, "Show the columns of the dataset")
pandas_ai.run(df, "Show the data types of the columns")
pandas_ai.run(df, "Show the summary statistics of the dataset")
pandas_ai.run(df, "Show the missing values in the dataset")
pandas_ai.run(df, "Show the duplicate rows in the dataset")
pandas_ai.run(df, "Drop the duplicate rows in the dataset")
pandas_ai.run(df, "Show the unique values in the 'Outlet' column")
pandas_ai.run(df, "Show the unique values in the 'Product' column")
pandas_ai.run(df, "Show the total sales for each product")
pandas_ai.run(df, "Show the total sales for each outlet")
pandas_ai.run(df, "Plot the chart of the products based on total_sales")
pandas_ai.run(df, "Plot the bar chart of the products  based on the monthly sales")
pandas_ai.run(df, "Plot the pie chart of the overall  products sales")