import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

# Create a sample DataFrame


df = pd.read_csv("amazon.csv");

# Handling Missing Values
def check_missing_values(dataframe):
    return dataframe.isnull().sum()


print(check_missing_values(df))
df[df.rating_count.isnull()]


# Removing NaN values from thr rowsa
df.dropna(subset=["rating_count"], inplace=True)
print(check_missing_values(df))


# Check for the duplicate values
def check_duplicates(dataframe):
    return dataframe.duplicated().sum()


print(check_duplicates(df))


# Check the datatyp0es
def check_data_types(dataframe):
    return dataframe.dtypes


print(check_data_types(df))


# Some variables in a dataset may have an object data type, which means they are strings. In order to perform numerical analysis on these variables, we need to convert them to numeric values. For example, if we want to calculate the total price of all products, we cannot do so if the price variable is in object format. We need to convert it to a numeric data type first.
df["discounted_price"] = (
    df["discounted_price"]
    .astype(str)
    .str.replace("₹", "")
    .str.replace(",", "")
    .astype(float)
)
df["actual_price"] = (
    df["actual_price"]
    .astype(str)
    .str.replace("₹", "")
    .str.replace(",", "")
    .astype(float)
)
df["discount_percentage"] = (
    df["discount_percentage"].astype(str).str.replace("%", "").astype(float) / 100
)


# The rating column has a value with an incorrect character, so we will exclude
# the row to obtain a clean dataset.
count = df["rating"].str.contains("\|").sum()
print(f"Total de linhas com '|' na coluna 'rating': {count}")
df = df[df["rating"].apply(lambda x: "|" not in str(x))]
count = df["rating"].str.contains("\|").sum()
print(f"Total de linhas com '|' na coluna 'rating': {count}")


df["rating"] = df["rating"].astype(str).str.replace(",", "").astype(float)
df["rating_count"] = df["rating_count"].astype(str).str.replace(",", "").astype(float)


print(check_data_types(df))


# Now that we have adjusted the data types, let's create one more colum that could be interesting to have in our database. "rating_weighted" because it can be created as a way of considering not only the average rating, but also the number of people who rated the product. This column weighs the average rating by the number of ratings, giving more weight to ratings with a large number of raters. This can help identify products with high customer satisfaction and many positive ratings compared to products with high average ratings but few raters.
df["rating_weighted"] = df["rating"] * df["rating_count"]


df["sub_category"] = df["category"].astype(str).str.split("|").str[-1]
df["main_category"] = df["category"].astype(str).str.split("|").str[0]


df.columns

len(df)
df.head()


# Convert prices to numeric values
# df['discounted_price'] = df['discounted_price'].replace('[₹,]', '', regex=True).astype(float)
# df['actual_price'] = df['actual_price'].replace('[₹,]', '', regex=True).astype(float)
# df['rating_count'] = df['rating_count'].replace('[,]', '', regex=True).astype(int)

# # Select features for clustering
# features = df[['discounted_price', 'rating']]

# # Standardize the features
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)

# # Apply KMeans clustering
# kmeans = KMeans(n_clusters=2, random_state=42)
# df['cluster'] = kmeans.fit_predict(features_scaled)

# # Title of the app
# st.title('Amazon Product Recommendation System')

# # Select a product
# selected_product = st.selectbox('Select a product:', df['product_name'])

# # Find the cluster of the selected product
# selected_cluster = df[df['product_name'] == selected_product]['cluster'].values[0]

# # Recommend products from the same cluster
# recommended_products = df[df['cluster'] == selected_cluster]

# # Display the recommended products
# st.write('Recommended Products:')
# st.dataframe(recommended_products)
