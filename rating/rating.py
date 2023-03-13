##############################################
# Rating Products
##############################################

# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating

# !!! 'social proof'
##############################################
# Uygulama: Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama
##############################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: '%.5f' % x)

# (50+ Saat) Python A-Z: Veri Bilimi ve Machine Learning
# Puan: 4.8 (4.764925)
# Toplam Puan: 4611
# Puan Yüzdeleri: 75, 20, 4, 1, <1
# Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6


df = pd.read_csv("datasets/course_reviews.csv")
df.head()
df.shape

# rating dağılımı
df["Rating"].value_counts()

df["Questions Asked"].value_counts()

df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating": "mean"})

####################
# Average
####################

# Ortalama Puan
df["Rating"].mean()

####################
# Time-Based Weighted Average
####################
# Puan Zamanlarına Göre Ağırlıklı Ortalama

df.head()
df.info()

df["Timestamp"] = pd.to_datetime(df["Timestamp"])   # zaman değişkenine dönüştürür
df.info()

current_date = pd.to_datetime('2021-02-10 0:0:0')

df["days"] = (current_date - df["Timestamp"]).dt.days  # ne zaman önce yorum yapılmış
df.head()

df[df["days"] <= 30].count()   # son 30 günde kaç tane yorum yapılmış?

df.loc[df["days"] <= 30, "Rating"].mean()   # son 30 günde yapılan yorumların ortalaması

df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()    # bu aralıktaki yorum oranı

df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()    # bu aralıktaki yorum oranı

df.loc[df["days"] > 180, "Rating"].mean()


df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
    df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
    df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
    df.loc[df["days"] > 180, "Rating"].mean() * 22/100

# sürekli kullanabileceğimiz için fonksiyonlaştırma yapalım
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return  dataframe.loc[dataframe["days"] <= 30, "Rating"].mean() * w1 / 100 + \
            dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
            dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
            dataframe.loc[dataframe["days"] > 180, "Rating"].mean() * w4 / 100

time_based_weighted_average(df)

time_based_weighted_average(df, 30, 26, 22, 22)


####################
# User-Based Weighted Average
####################

df.head()

df.groupby("Progress").agg({"Rating": "mean"})  # izlenme oranına bakılarak verilen puan analizi

df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[df["Progress"] > 75, "Rating"].mean() * 28 / 100

# fonksiyonlaştırma
def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return  dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
            dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
            dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
            dataframe.loc[dataframe["Progress"] > 75, "Rating"].mean() * w4 / 100

user_based_weighted_average(df, 20, 24, 26, 30)


####################
# Weighted Rating
####################
# Bu iki fonksiyonu bir araya getirerek daha kaliteli, daha güven verici ve daha hassas bir ortalama hesabı gerçekleştirmiş olduk.

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w / 100 + user_based_weighted_average(dataframe) * user_w / 100

course_weighted_rating(df)

course_weighted_rating(df, time_w=40, user_w=60)