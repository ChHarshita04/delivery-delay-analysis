import pandas as pd

# --------------------------------------------------
# 1. DATA LOADING
# --------------------------------------------------

df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding="latin1")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. INITIAL DATA INSPECTION
# --------------------------------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)


# --------------------------------------------------
# 3. CHECK FOR MISSING VALUES
# --------------------------------------------------

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nTotal missing values:")
print(df.isnull().sum().sum())


# --------------------------------------------------
# 4. HANDLE MISSING VALUES
# --------------------------------------------------

# Identify numerical and categorical columns
numerical_columns = df.select_dtypes(include="number").columns
categorical_columns = df.select_dtypes(include="object").columns

# Fill missing numerical values with median
for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill missing categorical values with "Unknown"
for column in categorical_columns:
    df[column] = df[column].fillna("Unknown")

print("\nMissing values after handling:")
print(df.isnull().sum().sum())


# --------------------------------------------------
# 5. CHECK AND REMOVE DUPLICATE RECORDS
# --------------------------------------------------

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("Dataset shape after removing duplicates:")
print(df.shape)


# --------------------------------------------------
# 6. HANDLE INCONSISTENT TEXT FORMATS
# --------------------------------------------------

# Remove unnecessary spaces from text columns
for column in categorical_columns:
    df[column] = df[column].str.strip()

# Standardize Shipping Mode values
if "Shipping Mode" in df.columns:
    df["Shipping Mode"] = df["Shipping Mode"].str.title()

print("\nShipping Mode values:")
print(df["Shipping Mode"].unique())


# --------------------------------------------------
# 7. CHECK FOR INVALID VALUES
# --------------------------------------------------

# Shipping days should not be negative
if "Days for shipping (real)" in df.columns:
    invalid_shipping_days = (df["Days for shipping (real)"] < 0).sum()
    print("\nInvalid negative shipping days:", invalid_shipping_days)

# Order quantity should not be negative
if "Order Item Quantity" in df.columns:
    invalid_quantity = (df["Order Item Quantity"] < 0).sum()
    print("Invalid negative order quantities:", invalid_quantity)

# Sales should not normally be negative
if "Sales" in df.columns:
    invalid_sales = (df["Sales"] < 0).sum()
    print("Invalid negative sales values:", invalid_sales)


# --------------------------------------------------
# 8. DATA TYPE CONVERSION
# --------------------------------------------------

# Convert order date to datetime
if "order date (DateOrders)" in df.columns:
    df["order date (DateOrders)"] = pd.to_datetime(
        df["order date (DateOrders)"],
        errors="coerce"
    )

# Convert shipping date to datetime
if "shipping date (DateOrders)" in df.columns:
    df["shipping date (DateOrders)"] = pd.to_datetime(
        df["shipping date (DateOrders)"],
        errors="coerce"
    )

print("\nData types after conversion:")
print(df[[
    "order date (DateOrders)",
    "shipping date (DateOrders)"
]].dtypes)


# --------------------------------------------------
# 9. DATA NORMALIZATION
# --------------------------------------------------

# Min-Max normalization
# Formula:
# (X - Minimum) / (Maximum - Minimum)

columns_to_normalize = [
    "Sales",
    "Order Item Quantity",
    "Days for shipping (real)"
]

for column in columns_to_normalize:
    if column in df.columns:
        minimum = df[column].min()
        maximum = df[column].max()

        if maximum != minimum:
            df[column + "_Normalized"] = (
                (df[column] - minimum) / (maximum - minimum)
            )

print("\nNormalized columns created:")
print([
    column for column in df.columns
    if column.endswith("_Normalized")
])


# --------------------------------------------------
# 10. FINAL DATA VALIDATION
# --------------------------------------------------

print("\nFinal dataset shape:")
print(df.shape)

print("\nRemaining missing values:")
print(df.isnull().sum().sum())

print("\nFinal data types:")
print(df.dtypes)

print("\nFirst 5 rows of cleaned dataset:")
print(df.head())


# --------------------------------------------------
# 11. SAVE CLEANED DATASET
# --------------------------------------------------

df.to_csv("cleaned_DataCoSupplyChainDataset.csv", index=False)

print("\nCleaned dataset saved successfully!")