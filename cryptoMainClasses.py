from pycoingecko import CoinGeckoAPI
from datetime import date , timedelta
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

cg=CoinGeckoAPI()
coin_list=['bitcoin','ethereum','tether','ripple','binancecoin','solana','usd-coin','dogecoin','cardano','tron']

class cryptix:
    def __init__(self,name):
        self.name = name
        eth_result = cg.get_coin_ohlc_by_id(id =self.name, vs_currency = "usd", days = "365")
        eth_pred = cg.get_coin_ohlc_by_id(id =self.name, vs_currency = "usd", days = "1")

        Date = date.today() + timedelta(days=1)

        df = pd.DataFrame(eth_result)
        df.columns = ["date", "open", "high", "low", "close"]

        df_pred = pd.DataFrame(eth_pred)
        df_pred.columns = ["date", "open", "high", "low", "close"]

        X=pd.to_datetime(df["date"], unit = "ms")
        Y=df['close'].to_numpy()
        #print(X)

        df["date"] = pd.to_datetime(df["date"], unit = "ms")
        df.set_index('date', inplace = True)

        df['open-close']=df['open']-df['close']
        df['high-low']=df['high']-df['low']

        df_pred['open-close']=df_pred['open']-df_pred['close']
        df_pred['high-low']=df_pred['high']-df_pred['low']
        df_pred["date"] = pd.to_datetime(df_pred["date"], unit = "ms")
        df_pred.set_index('date', inplace = True)

        x=df[['high-low','open-close']].to_numpy()

        c=0
        c1=0
        c2=0

        m=Y
        n=[]#storing values 1 or -1
        n.append(-1)
        for i in range(1,len(df)):
            if m[i] > m[i-1]:
                n.append(1)
            else :
                n.append(-1)


        y=n.copy()
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=44)
        sc_x=StandardScaler()
        x_train = sc_x.fit_transform(x_train)
        x_test=sc_x.transform(x_test)

        #using grid search to find the best parameter
        k={'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12]}
        knn=KNeighborsClassifier()
        model=GridSearchCV(knn,k,cv=5)

        model.fit(x_train,y_train)

        x_pred=df_pred[['high-low','open-close']].to_numpy()
        y_pred=model.predict(x_pred)

        for c in range(len(y_pred)):
            if y_pred[c]==1:
                c1=c1+1
            else:
                c2=c2+1

        if(c1>c2):
            prob=(c1/len(y_pred))
            print("There is ",prob*100,"% of increasing of value and would be better if you buy")
        else:
             prob=(c2/len(y_pred))
             print("There is ",prob*100,"% of decreasing of value and would be better if you sell or don't buy")


a=cryptix(coin_list[4])