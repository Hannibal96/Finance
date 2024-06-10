import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

symbol_name_convertor = {
    '^GSPC': 'S&P 500',
    '^DJI': 'Dow Jones Industrial Average',
    '^IXIC': 'NASDAQ Composite',
    '^FTSE': 'FTSE 100',
    '^GDAXI': 'DAX',
    '^FCHI': 'CAC 40',
    '^N225': 'Nikkei 225',
    '^HSI': 'Hang Seng Index',
    '000001.SS': 'Shanghai Composite',
    '^BSESN': 'BSE Sensex'
}

name_symbol_convertor = {v: k for k, v in symbol_name_convertor.items()}


class DataStock:
    def __init__(self, symbol: str = '^GSPC', start_date: str = '1999-01-01', end_date:str = '2024-01-01'):
        self.symbol = symbol
        self.stock = yf.download(symbol, start=start_date, end=end_date)
        self.stock_values = self.stock['Close'].to_numpy()
        self.stock_dates = self.stock.index
        self.total_days = (self.stock_dates[-1] - self.stock_dates[0]).days
        self.gain = self.stock_values[-1] / self.stock_values[0]
        self.alpha_month = self.gain ** (1 / (12 * (self.total_days / 365)))

    def get_data(self):
        return self.stock_values

    def plot_data(self):
        plt.plot(self.stock_dates, self.stock_values)
        plt.grid()
        plt.title(symbol_name_convertor[self.symbol])
        plt.show()


if __name__ == "__main__":
    current_date = datetime.today().strftime('%Y-%m-%d')
    data = DataStock()
    data.plot_data()





