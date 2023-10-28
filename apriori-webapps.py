import streamlit as st
import pandas as pd
# import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# st.title('Aplikasi Web Data Mining')
st.write("""
# Aplikasi Web Data Mining untuk Analisis Data Toko Bangunan Menggunakan Algoritma Apriori
Aplikasi berbasis web yang menampilkan aturan asosiasi yang terbentuk dari data penjualan toko bangunan
""")

st.sidebar.header('Parameter Inputan')

#upload file
upload_file = st.sidebar.file_uploader("Upload file excel", type=['xlsx'])
if upload_file is not None:
    data = pd.read_excel(upload_file)
else:
    #load data
    data = pd.read_excel("dataset_tb.xlsx")

st.markdown("## Data Penjualan")
st.write(data)

support = st.sidebar.selectbox('Min Support', (0.01,0.02,0.03,0.04,0.05))

#CATEGORY
# Stripping extra spaces in the description
data['Kategori'] = data['Kategori'].str.strip()
  
# Dropping the rows without any invoice number
data.dropna(axis = 0, subset =['ID Transaksi'], inplace = True)
data['ID Transaksi'] = data['ID Transaksi'].astype('str')
  
basket_tb = (data.groupby(['ID Transaksi', 'Kategori'])['Jumlah_pembulatan']
          .sum().unstack().reset_index().fillna(0)
          .set_index('ID Transaksi'))

def hot_encode(x):
    if(x<= 0):
        return 0
    if(x>= 1):
        return 1

data_hot_encode = basket_tb.applymap(hot_encode)
basket_tb = data_hot_encode

# Building the model
frq_items = apriori(basket_tb, min_support = support, use_colnames = True)

# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
st.markdown("## Aturan Asosiasi")
st.write(rules)

sns.scatterplot(x = "support", y = "confidence", data = rules)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("## Persebaran Support dan Confidence")
st.pyplot()

filtered_rules = rules[(rules['confidence'] >= 0.5) &
                        (rules['lift'] > 1.0)]

# take the "antecedents" and "consequents" values
antecedents2 = list(filtered_rules["antecedents"].apply(lambda x: list(x))) # convert them to list with lambda function in Pandas.apply()
consequents2 = list(filtered_rules["consequents"].apply(lambda x: list(x)))

st.markdown("## Hasil Rekomendasi:")
for item in antecedents2:
    idx = antecedents2.index(item)
    if len(item) == 1:
        st.success(f"Jika konsumen membeli **{item[0]}**, maka membeli **{consequents2[idx][0]}** secara bersamaan")
    elif len(item) == 2:
        st.success(f"Jika konsumen membeli **{item[0]}** dan **{item[1]}**, maka membeli **{consequents2[idx][0]}** secara bersamaan")