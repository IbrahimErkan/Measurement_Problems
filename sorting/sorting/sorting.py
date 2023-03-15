##############################################
# Sorting Product
##############################################

##############################################
# Uygulama: Kurs Sıralama
##############################################

import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # Standartlaştırma

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: '%.5f' % x)

df = pd.read_csv("datasets/product_sorting.csv")
df.shape
df.head()


#####################
# Sorting by Rating
#####################

df.sort_values("rating", ascending=False).head(20)


#####################
# Sorting by Comment Count or Purchase Count
#####################

df.sort_values("purchase_count", ascending=False).head(20)

df.sort_values("commment_count", ascending=False).head(20)


#####################
# Sorting by Rating, Commment and Purchase
#####################

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

df.head()
df.describe().T


df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])

df.head()
df.describe().T

# ağırlıklandır
(df["comment_count_scaled"] * 32 / 100 +
 df["purchase_count_scaled"] * 26 / 100 +
 df["rating"] * 42 / 100)


# fonksiyonlaştırma
def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["comment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)

weighted_sorting_score(df)

df["weighted_sorting_score"] = weighted_sorting_score(df)

df.sort_values("weighted_sorting_score", ascending=False).head()

df[df["course_name"].str.contains("Veri Bilimi")].\
    sort_values("weighted_sorting_score", ascending=False).head(20)   # kurs isimlerinde "Veri Bilimi" geçenleri getir
