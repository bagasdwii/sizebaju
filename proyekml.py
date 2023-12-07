import pickle
import streamlit as st
import pandas as pd
import os
import numpy as np
import altair as alt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

model = pickle.load(open('DecisionTree_Classifier.sav', 'rb'))

st.title('Prediksi Ukuran Baju')
st.image('./baju.jpg')
st.header("Dataset")
#open file xlsx
df1 = pd.read_csv('final_test_10k.csv')
st.dataframe(df1)

st.write("Grafik Weight")
chart_weight = pd.DataFrame(df1, columns=["weight"])
st.line_chart(chart_weight)
st.write("Grafik Weight")
chart_weight = pd.DataFrame(df1, columns=["weight"])
st.bar_chart(chart_weight)
st.write("Grafik Weight")
chart_weight = pd.DataFrame(df1, columns=["weight"])
st.bar_chart(chart_weight)

st.write("Grafik Age")
chart_age = pd.DataFrame(df1, columns=["age"])
st.line_chart(chart_age)
st.write("Grafik Age")
chart_age = pd.DataFrame(df1, columns=["age"])
st.bar_chart(chart_age)



st.title('Grafik Size')

# Visualisasi menggunakan Altair
chart = alt.Chart(df1).mark_bar().encode(
    x='size:N',
    y='count()'
).properties(
    width=600,
    height=400
)

st.altair_chart(chart, use_container_width=True)


#input nilai dari variable independent
weight = st.number_input("Weight ", 0, 200)
height = st.number_input("`Height` ", 0, 500)
age = st.number_input("Age ", 0, 150)

if st.button('Prediksi'):
    # Membuat array numpy dari nilai yang dimasukkan
    input_data = np.array([[weight, height, age]])

    # Melakukan prediksi menggunakan model Decision Tree
    predicted_size = model.predict(input_data)
    st.write (predicted_size)
    st.write(f"Prediksi Ukuran Baju: {predicted_size[0]}")

    