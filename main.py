import os
import requests
import logging
import json
import argparse
from datetime import datetime, timedelta
import multiprocessing
import pandas as pd
import numpy as np
import ta

from stock_twits.constants import (
    INCREMENT_MESSAGES,
    SELECTED_MARKET,
    STARTING_HISTORICAL_DATE,
    STARTING_INTERNVAL,
    WATCHLIST_INTERVALS,
    PROXIES,
    HEADERS,
    POLYGON_API_KEY,  
    OPENAI_API_KEY,  
)
from stock_twits.extract import (
    start_historical_generation,
    start_real_generation,
    start_watchlist_generation,
)
from stock_twits.utils import (
    convert_to_timezone,
    create_logger,
    extract_hour_day_minute,
    generate_months,
    generate_time_intervals_24_hour_format,
    market_open,
)

pd.set_option("future.no_silent_downcasting", True)
INTERVALS = generate_time_intervals_24_hour_format(STARTING_INTERNVAL)
logger = create_logger(__name__, os.environ["LOG_LEVEL"])

def fetch_sentiment_data(ticker, end_time):
    url = f"https://api-gw-prd.stocktwits.com/sentiment-api/{ticker}/detail?end={end_time}"
    try:
        response = requests.get(url, headers=HEADERS, proxies=PROXIES)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        logger.error(f"Other error occurred: {err}")
    return None

def fetch_news_from_polygon(ticker):
    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&apiKey={POLYGON_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get('results', [])
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        logger.error(f"Other error occurred: {err}")
    return []

def get_embeddings_from_chatgpt(text):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "text-embedding-ada-002",
        "input": text
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get('data', [])[0].get('embedding', [])
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        logger.error(f"Other error occurred: {err}")
    return []

def fetch_price_volume_data(ticker):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2020-01-01/2023-01-01?apiKey={POLYGON_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get('results', [])
        if not data:
            return pd.DataFrame()
        
        # Use NumPy for faster data manipulation
        timestamps = np.array([item['t'] for item in data], dtype='datetime64[ms]')
        close_prices = np.array([item['c'] for item in data])
        volumes = np.array([item['v'] for item in data])
        
        df = pd.DataFrame({
            't': timestamps,
            'c': close_prices,
            'v': volumes
        })
        df.set_index('t', inplace=True)
        return df
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        logger.error(f"Other error occurred: {err}")
    return pd.DataFrame()

def add_technical_indicators(df):
    close_prices = df['c'].values
    volumes = df['v'].values
    
    # Calculate SMA using NumPy
    window = 20
    sma = np.convolve(close_prices, np.ones(window), 'valid') / window
    df['SMA'] = np.concatenate((np.full(window-1, np.nan), sma))
    
    # Calculate EMA using NumPy
    ema = np.zeros_like(close_prices)
    alpha = 2 / (window + 1)
    ema[0] = close_prices[0]
    for i in range(1, len(close_prices)):
        ema[i] = alpha * close_prices[i] + (1 - alpha) * ema[i-1]
    df['EMA'] = ema
    
    # Calculate RSI using NumPy
    window = 14
    deltas = np.diff(close_prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    rs = up / down
    rsi = np.zeros_like(close_prices)
    rsi[:window] = 100. - 100. / (1. + rs)
    
    for i in range(window, len(close_prices)):
        delta = deltas[i - 1]  # The diff is 1 shorter
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up * (window - 1) + upval) / window
        down = (down * (window - 1) + downval) / window
        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)
    df['RSI'] = rsi
    
    # Calculate MACD using NumPy
    short_window = 12
    long_window = 26
    signal_window = 9
    
    short_ema = np.zeros_like(close_prices)
    long_ema = np.zeros_like(close_prices)
    macd = np.zeros_like(close_prices)
    signal = np.zeros_like(close_prices)
    
    short_alpha = 2 / (short_window + 1)
    long_alpha = 2 / (long_window + 1)
    
    short_ema[0] = close_prices[0]
    long_ema[0] = close_prices[0]
    
    for i in range(1, len(close_prices)):
        short_ema[i] = short_alpha * close_prices[i] + (1 - short_alpha) * short_ema[i-1]
        long_ema[i] = long_alpha * close_prices[i] + (1 - long_alpha) * long_ema[i-1]
        macd[i] = short_ema[i] - long_ema[i]
    
    signal[0] = macd[0]
    signal_alpha = 2 / (signal_window + 1)
    
    for i in range(1, len(macd)):
        signal[i] = signal_alpha * macd[i] + (1 - signal_alpha) * signal[i-1]
    
    df['MACD'] = macd
    df['MACD_SIGNAL'] = signal
    df['MACD_DIFF'] = macd - signal
    
    return df

def fetch_fundamental_data(ticker: str) -> None:
    """
    Generates simulated financial data for a given ticker and saves it to CSV files.
    """
    # Generate dates for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create a DataFrame with dates
    df = pd.DataFrame({'Date': dates})

    # Generate simulated financial data
    np.random.seed(42)  # For reproducibility
    df['netCashFlow'] = np.random.uniform(-1e9, 1e9, len(df))
    df['net_income_loss'] = np.random.uniform(-1e9, 1e9, len(df))
    df['operating_expenses'] = np.random.uniform(0, 1e9, len(df))
    df['revenues'] = np.random.uniform(0, 2e9, len(df))
    df['basic_earnings_per_share'] = np.random.uniform(-10, 10, len(df))

    # Save financial data to separate CSV files for each day
    for date in dates:
        year_month = date.strftime("%Y-%m")
        day = date.strftime("%d")
        directory = f"data/hfprices/{ticker}/{year_month}"
        os.makedirs(directory, exist_ok=True)
        csv_path = f"{directory}/{day}.csv"
        
        # Filter data for the specific date
        daily_data = df[df['Date'] == date]
        
        # Save data to CSV
        daily_data.to_csv(csv_path, index=False)
        
        print(f"Financial data for {date.date()} saved to {csv_path}")

    print(f"Generated and saved financial data for ticker {ticker}")

def save_to_csv(df, ticker, date):
    base_directory = "/Users/haroonwajid/Downloads/Stock-Data-Generator-main 2/data/hfprices"
    directory = f"{base_directory}/{ticker}/{date}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(f"{directory}/data.csv")

def save_fundamental_data(data, ticker, date):
    base_directory = "/Users/haroonwajid/Downloads/Stock-Data-Generator-main 2/data/hfprices"
    directory = f"{base_directory}/{ticker}/{date}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(f"{directory}/fundamentals.json", "w") as f:
        json.dump(data, f)

def save_slices(df, ticker, date, time):
    base_directory = "/Users/haroonwajid/Downloads/Stock-Data-Generator-main 2/data/hfprices"
    directory = f"{base_directory}/{ticker}/29dayslices/{date}/{time}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(f"{directory}/data.csv")

def main():
    parser = argparse.ArgumentParser(
        description="Extract stock data in real-time or historical."
    )
    parser.add_argument(
        "--type",
        choices=["real", "history", "watchlist", "news", "indicators", "fundamentals"],  # Add "fundamentals" option
        required=True,
        help="Specify the type of data to extract: 'real' for real-time data, 'history' for historical data, 'watchlist' for watchlist generation, 'news' for news articles, 'indicators' for technical indicators, or 'fundamentals' for fundamental data.",
    )
    parser.add_argument(
        "--day",
        required=False,
        help="""Specify the day for which you want to run the historical or watchlist generator in format `%Y-%m-%d`. It will run the generator only for that date""",
    )
    parser.add_argument(
        "--all",
        required=False,
        choices=["y", "n"],
        default="y",
        help="If you want to run the generator from start of `2023-01-01`",
    )
    parser.add_argument(
        "--ticker",
        required=False,
        help="Specify the ticker for which you want to fetch news articles, indicators, or fundamentals.",
    )
    args = parser.parse_args()
    if args.type == "history":
        _all = args.all
        if _all == "y":  # start historical generation from start of the `2023-01-01`
            for current_day in generate_months(STARTING_HISTORICAL_DATE):
                if market_open(SELECTED_MARKET, current_day):
                    logger.info(f"Market was opened on {current_day}")
                    start_historical_generation(
                        current_day, INTERVALS, INCREMENT_MESSAGES
                    )
                else:
                    logger.info(f"Market was closed on {current_day}")
        elif _all == "n" and args.day:
            if market_open(SELECTED_MARKET, args.day):
                logger.info(f"Market was opened on {args.day}")
                start_historical_generation(args.day, INTERVALS, INCREMENT_MESSAGES)
            else:
                logger.info(f"Market was closed on {args.day}")
        else:
            logger.error("You need to select a `day` if `all` is n")
    elif args.type == "watchlist":
        _all = args.all
        if _all == "y":  # start watchlist generation from start of the `2023-01-01`
            for current_day in generate_months(STARTING_HISTORICAL_DATE):
                if market_open(SELECTED_MARKET, current_day):
                    start_watchlist_generation(current_day, WATCHLIST_INTERVALS)
                else:
                    logger.info(f"Market was closed on {current_day}")
        elif _all == "n" and args.day:
            if market_open(SELECTED_MARKET, args.day):
                start_watchlist_generation(args.day, WATCHLIST_INTERVALS)
            else:
                logger.info(f"Market was closed on {args.day}")
        else:
            logger.error("You need to select a `day` if `all` is n")

    elif args.type == "real":  ## starting the real-time server for getting the data
        # need to make sure that extraction is only hit when market is active.
        logger.info("Starting the real-time server for Ticker Monitoring")
        while True:
            today = datetime.strftime(datetime.now(), "%Y-%m-%d")
            logger.info(f"Checking if {SELECTED_MARKET} Market is opened on {today}")
            if market_open(SELECTED_MARKET, today):
                logger.info(f"{SELECTED_MARKET} Market is opened on {today}")
                timestamp = convert_to_timezone(datetime.now())
                current_time_dict = extract_hour_day_minute(timestamp)
                logger.info(f"We are not at in one of our 72 intervals: {timestamp}")
                if (
                    timestamp.microsecond == 0
                    and timestamp.second == 0
                    and current_time_dict["hour"] + current_time_dict["minute"]
                    in INTERVALS["total"]
                ):  # As soon as our timestamp in one of our 72 interval correct to microseconds
                    # this make sure that our job runs 72 times each day
                    logger.info(
                        f"Extracting the Messages as we are at 10 minute mark: {timestamp}"
                    )
                    process = multiprocessing.Process(
                        target=start_real_generation,
                        args=(timestamp,),
                    )
                    process.start()
            else:
                logger.info(
                    f"{SELECTED_MARKET} Market is closed on {today}. Therefore cannot start monitoring"
                )
                break

    elif args.type == "news":  # Add news option handling
        if not args.ticker:
            logger.error("You need to specify a `ticker` for fetching news articles")
            return
        news_articles = fetch_news_from_polygon(args.ticker)
        for article in news_articles:
            content = article.get("description", "")
            if content:
                embedding = get_embeddings_from_chatgpt(content)
                logger.info(f"Fetched embedding for article: {article.get('title', '')}")

    elif args.type == "indicators":  # Add indicators option handling
        if not args.ticker:
            logger.error("You need to specify a `ticker` for fetching indicators")
            return
        df = fetch_price_volume_data(args.ticker)
        if not df.empty:
            df = add_technical_indicators(df)
            logger.info(f"Added technical indicators for {args.ticker}")
            date = datetime.now().strftime("%Y-%m-%d")
            save_to_csv(df, args.ticker, date)
        else:
            logger.error(f"Failed to fetch price/volume data for {args.ticker}")

    elif args.type == "fundamentals":  # Add fundamentals option handling
        if not args.ticker:
            logger.error("You need to specify a `ticker` for fetching fundamentals")
            return
        fetch_fundamental_data(args.ticker)

if __name__ == "__main__":
    main()
