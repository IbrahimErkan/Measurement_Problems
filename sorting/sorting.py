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

