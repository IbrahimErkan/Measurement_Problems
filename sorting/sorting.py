##############################################
# Sorting Product
##############################################

##############################################
# Uygulama: Kurs Sıralama
##############################################
import math
import scipy.stats as st
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


#####################
# Bayesian Average Rating Score
#####################
# not: Bu yöntem her bir kurs için ayrı ayrı olan puanların dağılım bilgisini kullanarak bize bir average rating ortalama puan hesabı yapacak.


# sorting Product with 5 Star Rated          # 5 Yıldız Derecelendirmeli Ürün sıralama
# Sorting Product According to Distribution of 5 Star Rating        # Ürünü 5 Yıldız Puanı Dağılımı'na Göre Sıralama

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

df.head()

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)

df.sort_values("weighted_sorting_score", ascending=False).head(20)
df.sort_values("bar_score", ascending=False).head(20)

df[df["course_name"].index.isin([5, 1])]      # kurs isimlenrinden indexlere göre 5. ve 1. indexdeki kurslara getir.

df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False)


def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return bar_score*bar_w/100 + wss_score*wss_w/100


df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values("hybrid_sorting_score", ascending=False).head(20)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)


#############################
# Uygulama: IMDB Movie Scoring & Sorting
#############################

import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("/content/gdrive/MyDrive/DSMLBC10/week_5 (27.10.22-02.11.22)/datasets/movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin

df = df[["title", "vote_average", "vote_count"]]

print(df.shape)
df.head()


# Vote Average'a Göre Sıralama
df.sort_values("vote_average", ascending=False).head()

df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T


df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head(20)

#tercih olarak 400 den büyük seçildi. Bu yemedi. vote_count 1-10 arasında scale edip değerlendirmek daha mantıklı olabilir.



from sklearn.preprocessing import MinMaxScaler

df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)). \
    fit(df[["vote_count"]]). \
    transform(df[["vote_count"]])


# vote_average * vote_count (ikiside 1-10 arası değer)

df["average_count_score"] = df["vote_average"] * df["vote_count_score"]
df.sort_values("average_count_score", ascending=False).head(20)
