import pandas as pd

# Create a DataFrame with mock data
data = {
    "Ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
}

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("data/stocklista.csv", index=False)