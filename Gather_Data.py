import yfinance as yf

###
#Function takes input:
#   Ticker: Valid Ticker from Yahoo Finance
#   Period: (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)
#   Interval: (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo)
#
#Output:
#Pandas series consisting of close prices and dividends
#
#Note: Function can be edited to include 'Open', 'High', 'low', 'splits' and 'volume'
#

def GatherData(Ticker, Period, Interval):
    historicalData = yf.Ticker(Ticker).history(period = Period, interval = Interval)
    return historicalData[["Close", "Dividends"]]

#Testing
if __name__ == "__main__":
    data = GatherData("ORG.AX", "1y", "1d")
    print(data)
    print(type(data))

    data = GatherData("AAPL", "10y", "1mo")
    print(data)