import pandas as pd 
import numpy as np 
import requests
import json
import talib
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


def request (limit):
        limit = str(limit)
        url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=30m&limit="+limit
        payload={}
        headers = {'Content-Type': 'application/json'}
        response = requests.request("GET", url, headers=headers, data=payload)
        values = json.loads(response.text)

        dt = pd.DataFrame(values)
        dt = dt[[1,2,3,4,6,7]].round(2)

        dt.rename({1: 'Open', 2: 'High', 3: 'Low', 4: 'Close', 6: 'Timestamp', 7: 'Volume_(Currency)'}, axis=1, inplace=True)
        dt = round(dt.astype(float), 2)
        dt['Timestamp'] = dt['Timestamp'].astype(int)
        return dt



def indicators (dt):
    dtable = pd.DataFrame()
    dtable['atr'] = talib.ATR(dt['High'], dt['Low'], dt['Close'], timeperiod=100)
    dtable['ad'] = talib.AD(dt['High'], dt['Low'], dt['Close'], dt['Volume_(Currency)'])
    dtable['rsi'] = talib.RSI(dt['Close'], timeperiod=30)
    dtable['mfi'] = talib.MFI(dt['High'], dt['Low'], dt['Close'], dt['Volume_(Currency)'])
    dtable['macd'] = talib.MACD(dt['Close'], fastperiod=50, slowperiod=100, signalperiod=40)[0]
    dtable['obv'] = talib.OBV(dt['Close'], dt['Volume_(Currency)'])

    large2 = 240 #5d = 240 30m bars
    large3 = 480 #10d = 480 30m bars
    dtable['prev_5d_slope'] = round( ((dt['Close'] - dt['Close'].shift(large2)) / large2) ,3)
    dtable['prev_10d_slope'] = round( ((dt['Close'] - dt['Close'].shift(large3)) / large3) ,3)

    dtable['mfi2'] = dtable['mfi']
    dtable['rsi2'] = dtable['rsi']
    i = 0
    while(i < len(dtable)):
        if(dtable['prev_10d_slope'][i] >= 0):
            dtable['mfi2'][i] = dtable['mfi'][i]
            dtable['rsi2'][i] = dtable['rsi'][i]
        else:
            dtable['mfi2'][i] = ((dtable['mfi'][i]) * -1)
            dtable['rsi2'][i] = ((dtable['rsi'][i]) * -1)
        i += 1
    
    dtable = dtable[['atr', 'ad', 'prev_10d_slope', 'prev_5d_slope', 'obv', 'rsi2', 'mfi2', 'macd']]
    dtable = round(dtable, 1)
    dtable0 = dtable.iloc[-1:]        # present
    dtable1 = dtable.iloc[-49: -48]   # a day ago
    dtable2 = dtable.iloc[-97: -96]   # two days agoca
    dtable3 = dtable.iloc[-145: -144] # tree days ago
    dtable4 = dtable.iloc[-193: -192] # four days ago
    return dtable0, dtable1, dtable2, dtable3, dtable4



def run_model (dt):
    model = joblib.load('./models/model2.plk')
    prediction = model.predict(dt)
    return prediction




if __name__ == '__main__':

    dt0, dt1, dt2, dt3, dt4 = indicators(request(1000))

    fig, ax = plt.subplots(figsize=(14,8))

    t = request(240)
    x = (pd.to_datetime(t['Timestamp'], unit='ms', utc=None))
    y = t['Close']
    ax.plot(x, y, color='y')
    # yellow represent the past days


    y4 = (y[-1:] + run_model(dt4) * 240)
    y3 = (y[-49: -48] + run_model(dt3) * 240)
    y2 = (y[-97: -96] + run_model(dt2) * 240)
    y1 = (y[-145: -144] + run_model(dt1) * 240)
    y0 = (y[-193: -192] + run_model(dt0) * 240)

    def nextday(day):
        return (x[-1:]+timedelta(days=day))

    yy = [y[-1:], y4, y3, y2, y1, y0]
    xx = [nextday(0), nextday(1), nextday(2), nextday(3), nextday(4), nextday(5)]
    
    ax.plot(xx, yy, color='k', linewidth=2)
    # black represent the future days
    

    plt.title('Trend Prediction')
    plt.xlabel('Date')
    plt.ylabel('Bitcoin')
    plt.show()




