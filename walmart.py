import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Load data

FILE_NAME = "walmart_sales.csv"
df = pd.read_csv(FILE_NAME)

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Columns ===")
print(df.columns.tolist())

print("\n=== Info ===")
print(df.info())

print("\n=== Describe ===")
print(df.describe())


# Clean column names

df.columns = df.columns.str.strip()


# Clean + parse Date column

if "Date" not in df.columns:
    raise KeyError("Your dataset must contain a 'Date' column. Check df.columns output above.")

df["Date"] = (
    df["Date"]
    .astype(str)
    .str.strip()
    .str.replace(".", "/", regex=False)
    .str.replace("_", "/", regex=False)
)

# Parse with dayfirst=True to handle dd-mm-yyyy rows; coerce bad rows to NaT
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

bad_dates = df["Date"].isna().sum()
print(f"\nBad dates (could not parse): {bad_dates}")

# Drop rows where Date couldn't be parsed
df = df.dropna(subset=["Date"]).copy()


# Create time features

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)


#  Validate required columns exist
required_cols = ["Store", "Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}\nAvailable columns: {df.columns.tolist()}")

# Ensure numeric types (in case any columns came in as strings)
numeric_cols = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing numeric data (keeps analysis clean)
df = df.dropna(subset=numeric_cols).copy()


# Store performance

store_sales = df.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False)

print("\n=== Top 10 Stores by Total Sales ===")
print(store_sales.head(10))

plt.figure(figsize=(10, 5))
store_sales.head(10).plot(kind="bar")
plt.title("Top 10 Walmart Stores by Total Sales")
plt.ylabel("Total Sales")
plt.xlabel("Store")
plt.tight_layout()
plt.show()

# Holiday impact (average sales)
holiday_sales = df.groupby("Holiday_Flag")["Weekly_Sales"].mean()

print("\n=== Average Weekly Sales: Holiday vs Non-Holiday ===")
print(holiday_sales)

plt.figure(figsize=(6, 4))
holiday_sales.plot(kind="bar")
plt.title("Average Weekly Sales: Holiday vs Non-Holiday")
plt.ylabel("Average Weekly Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Add "holiday uplift %" as a nice business metric
if 0 in holiday_sales.index and 1 in holiday_sales.index:
    uplift_pct = ((holiday_sales.loc[1] - holiday_sales.loc[0]) / holiday_sales.loc[0]) * 100
    print(f"\nHoliday uplift: {uplift_pct:.2f}% (avg holiday week vs non-holiday week)")


# Weekly sales trend over time
weekly_sales_ts = df.groupby("Date")["Weekly_Sales"].sum().sort_index()

plt.figure(figsize=(12, 5))
plt.plot(weekly_sales_ts.index, weekly_sales_ts.values)
plt.title("Total Weekly Sales Trend (All Stores)")
plt.ylabel("Total Weekly Sales")
plt.xlabel("Date")
plt.tight_layout()
plt.show()


# Scatter plots (drivers)

plt.figure(figsize=(7, 5))
sns.scatterplot(x="Temperature", y="Weekly_Sales", data=df, alpha=0.4)
plt.title("Temperature vs Weekly Sales")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
sns.scatterplot(x="Unemployment", y="Weekly_Sales", data=df, alpha=0.4)
plt.title("Unemployment vs Weekly Sales")
plt.tight_layout()
plt.show()


# Correlation heatmap
corr = df[["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# Simple forecast (4-week moving average)

forecast_ma4 = weekly_sales_ts.rolling(window=4).mean()

plt.figure(figsize=(12, 5))
plt.plot(weekly_sales_ts.index, weekly_sales_ts.values, label="Actual")
plt.plot(forecast_ma4.index, forecast_ma4.values, label="4-week Moving Avg Forecast")
plt.title("Sales Forecast (4-Week Moving Average)")
plt.ylabel("Total Weekly Sales")
plt.xlabel("Date")
plt.legend()
plt.tight_layout()
plt.show()

print("\n✅ Script completed successfully.")
df.to_csv("walmart_clean.csv")