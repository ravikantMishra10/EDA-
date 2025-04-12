import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\ASUS\Desktop\Coffee Shop Sales.csv")

# 1. Data Overview using .info()
# Objective: Understand the structure of the dataset including data types and null values.
print("Objective 1. ")
print("Dataset Info:")
print(df.head())
df.info()




# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------



#  Descriptive Statistics using .describe()
# Objective: Summarize numerical features like total sales, price, and quantities.
print("Objective 2. ")
print("\nDescriptive Statistics:")
print(df.describe())



# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------




# 3. Outlier Detection with Boxplot
# Objective: Detect outliers in numerical columns like "Unit Price" or "Total Price".
print("Objective 3. \n")
# Boxplots for outlier detection
plt.figure(figsize=(14, 6))

# Unit Price
plt.subplot(1, 2, 1)
sns.boxplot(x=df['unit_price'], color='red')
plt.title('Outlier Detection: Unit Price')

# Total Price
plt.subplot(1, 2, 2)
sns.boxplot(x=df['total_price'], color='yellow')
plt.title('Outlier Detection: Total Price')

plt.tight_layout()
plt.show()



# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------




# 4. Unique Value Analysis
# Objective: Analyze uniqueness in categorical columns like Product Type, Branch, etc.
print("Objective 4. ")
print("\nUnique Values per Column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")



# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------

# 5. Visualization: Correlation Heatmap & Pairplot
# Objective: Show relationships between numerical features using heatmaps and pairplots


print("\nObjective 5. ")
# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='inferno')
plt.title('Correlation Heatmap')
plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------


# 6: Visual Analysis Using Multiple Charts
# Objective: Use a variety of visualizations (bar, pie, line, and heatmap) to extract business insights from different angles.

print("\nObjective 6. ")
# a. Bar Chart – Total Sales by Store Location
plt.figure(figsize=(8, 5))
sns.barplot(data=df.groupby('store_location')['total_price'].sum().reset_index(),
            x='store_location', y='total_price', palette='YlOrBr')
plt.title('Total Sales by Store Location')
plt.xlabel('Store Location')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()

# b. Pie Chart – Sales Distribution by Product Category
category_sales = df.groupby('product_category')['total_price'].sum()
plt.figure(figsize=(6, 6))
plt.pie(category_sales, labels=category_sales.index, autopct='%1.1f%%', startangle=140)
plt.title('Sales Distribution by Product Category')
plt.tight_layout()
plt.show()

# c. Line Plot – Quantity Sold Over Time
df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
df = df.dropna(subset=['transaction_date'])
daily_sales = df.groupby('transaction_date')['transaction_qty'].sum().reset_index()
daily_sales = daily_sales.sort_values('transaction_date')

plt.figure(figsize=(10, 5))
sns.lineplot(data=daily_sales, x='transaction_date', y='transaction_qty', color='green')
plt.title('Total Quantity Sold Over Time')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.tight_layout()
plt.show()


# d. Heatmap – Correlation Between Numeric Columns
plt.figure(figsize=(8, 5))
numeric_cols = df[['transaction_qty', 'unit_price', 'total_price']]
sns.heatmap(numeric_cols.corr(), annot=True, cmap='viridis')
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.show()


# e.this chart shows how frequently each product type was sold, giving a sense of what’s most popular.
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='product_type', order=df['product_type'].value_counts().index, palette='viridis')
plt.title('Frequency of Product Types Sold')
plt.xlabel('Product Type')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------



# 7. Data Summary with .sum() and .value_counts()
# Objective: Create summary stats like total sales by branch or payment method frequency.
print("\nObjective 7: Summary. ")

# 1. Total sales by store location
print("🔸 Total Sales by Store Location:")
print(df.groupby('store_location')['total_price'].sum())
print()

# 2. Total quantity sold by product category
print("🔸 Total Quantity Sold by Product Category:")
print(df.groupby('product_category')['transaction_qty'].sum().sort_values(ascending=False))
print()

# 3. Top 5 product types by total sales
print("🔸 Top 5 Product Types by Total Sales:")
print(df.groupby('product_type')['total_price'].sum().sort_values(ascending=False).head(5))
print()

# 4. Average unit price per product category
print("🔸 Average Unit Price per Product Category:")
print(df.groupby('product_category')['unit_price'].mean().sort_values(ascending=False))
print()

# 5. Most frequently purchased product details
print("🔸 Top 5 Most Frequently Purchased Products:")
print(df['product_detail'].value_counts().head(5))



