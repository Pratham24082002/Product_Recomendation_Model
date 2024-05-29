import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

# Create a sample DataFrame
df = pd.read_csv("amazon.csv")

# Handling Missing Values
df.dropna(subset=["rating_count"], inplace=True)

# Convert prices to numeric values
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

# Remove rows with incorrect rating values
df = df[df["rating"].apply(lambda x: "|" not in str(x))]

df["rating"] = df["rating"].astype(str).str.replace(",", "").astype(float)
df["rating_count"] = df["rating_count"].astype(str).str.replace(",", "").astype(float)

# Create weighted rating
df["rating_weighted"] = df["rating"] * df["rating_count"]

# Extract sub-category and main-category
df["sub_category"] = df["category"].astype(str).str.split("|").str[-1]
df["main_category"] = df["category"].astype(str).str.split("|").str[0]

# Select features for clustering
features = df[['discounted_price', 'rating']]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(features_scaled)

# Title of the app
st.title('Amazon Product Recommendation System')

# Autocomplete field for selecting a product
input_product = st.text_input('Start typing the product name:')
filtered_products = df[df['product_name'].str.contains(input_product, case=False, na=False)]

# Display autocomplete suggestions
if input_product:
    st.write("Suggestions:")
    suggestions = filtered_products['product_name'].unique()
    for suggestion in suggestions[:5]:  # Show up to 5 suggestions
        if st.button(suggestion):
            input_product = suggestion

# Find the cluster of the selected product if it exists in the DataFrame
if input_product in df['product_name'].values:
    selected_cluster = df[df['product_name'] == input_product]['cluster'].values[0]

    # Recommend products from the same cluster
    recommended_products = df[df['cluster'] == selected_cluster]

    # Display the recommended products as cards
    st.write('Recommended Products:')
    for _, row in recommended_products.iterrows():
        with st.expander(row['product_name']):
            st.write(f"**Category:** {row['category']}")
            st.write(f"**Discounted Price:** ₹{row['discounted_price']}")
            st.write(f"**Actual Price:** ₹{row['actual_price']}")
            st.write(f"**Discount Percentage:** {row['discount_percentage'] * 100}%")
            st.write(f"**Rating:** {row['rating']} ({row['rating_count']} ratings)")
            st.write(f"**About Product:** {row['about_product']}")
            st.image(row['img_link'])
            st.markdown(f"[Product Link]({row['product_link']})")
else:
    st.write("No product found with that name. Please try another product.")
