import argparse
import json
import configparser

import requests
import tvDatafeed


parser = argparse.ArgumentParser(prog = 'ETF Comparison')

# Download ETF tickers from ETFdb CSV
ETF_JSON_URL = 'https://raw.githubusercontent.com/JakubPluta/pyetf/refs/heads/main/etfpy/data/etfs/etfs_list.json'

# Read TradingView credentials from credentials.ini
config = configparser.ConfigParser()
config.read('credentials.ini')
username = config.get('DEFAULT', 'TRADINGVIEW_USERNAME')
password = config.get('DEFAULT', 'TRADINGVIEW_PASSWORD')


def download_tickers():
    print("Downloading ETF tickers from GitHub (JSON)...")
    response = requests.get(ETF_JSON_URL)
    etf_list = response.json()
    symbols = set(item['symbol'] for item in etf_list if 'symbol' in item)
    return symbols


# Download daily close prices using yahoo_fin
def download_stocks(tickers):
    exchanges = ['AMEX', 'NASDAQ', 'NYSE']

    print('Downloading data for %s tickers from TradingView' % len(tickers))
    tv = tvDatafeed.TvDatafeed(username, password)
    stocks = {}

    for i, ticker in enumerate(tickers, 1):
        print(f"Getting data for ({i}/{len(tickers)}): {ticker}")
        try:
            for exchange in exchanges:
                df = tv.get_hist(ticker, exchange=exchange, interval=tvDatafeed.Interval.in_weekly, n_bars=5000)
                if df is not None and not df.empty:
                    stock_output = [(str(row.name), row['close']) for _, row in df.iterrows()]
                    stocks[ticker] = stock_output
                    break
                else:
                    print(f'No data for {ticker} from TradingView.')
        except Exception as e:
            print(f'Error getting data for {ticker}: {e}')

    return stocks


def cache_stocks(stocks):
    print('Persisting data for %s tickers' % len(stocks))
    with open('etfs.json', 'w') as fp:
        json.dump(stocks, fp)


def main():
    args = parser.parse_args()
    tickers = download_tickers()
    stocks = download_stocks(tickers)
    cache_stocks(stocks)


if __name__ == "__main__":
    main()
