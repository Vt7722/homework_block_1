import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu
import regex

dataset = st.file_uploader("Загрузка датасета", type = ["csv"])
# df = pd.read_csv(dataset)
if dataset is not None:
    df = pd.read_csv(dataset)
    st.dataframe(df)
    cathegory=[]
    num=[]
    df2 = df
    for i in df.columns:
        df2[i] = df[i].astype(str).replace(",","", regex = True)
        if(regex.match("[+-]?[0-9]+\.[0-9]+",str(df[i][0])) or regex.match("^[0-9]+$", str(df[i][0]))):
            num.append(i)
        else:
            cathegory.append(i)
    st.dataframe(num)
    st.dataframe(cathegory)
    option1 = st.selectbox("Колонка 1", df.columns)
    #в зависимости от типа первой колонки определяются возможные варианты колонки 2
    if(option1 in num):
        option2 = st.selectbox("Колонка 2", num)
    else:
        option2 = st.selectbox("Колонка 2", cathegory)
    df[option1]=df[option1].astype(str).replace(",","", regex = True)
    df[option2]=df[option2].astype(str).replace(",","", regex = True)
    fig, ax = plt.subplots()
    ax = plt.hist(df[option1])
    plt.xticks(rotation = 90)
    st.pyplot(fig)
    fig, ax = plt.subplots()
    ax = plt.hist(df[option2])
    plt.xticks(rotation = 90)
    st.pyplot(fig)
    # в зависимости от типа колонки 1 разные варианты списков гипотез
    if(option1 in num):
        hip = st.selectbox("Алгоритм теста гипотез", ["t-test", "u-test", "chi-square"])
    else:
        hip = st.selectbox("Алгоритм теста гипотез", ["chi-square"])
    prediction=""  
    if st.button("Predict"):
        if(option1 != option2):
            if(hip == "t-test"):
                prediction=stats.ttest_ind(df[option1].astype(float).dropna(), df[option2].astype(float).dropna())
            if(hip == "u-test"):
                prediction=mannwhitneyu(df[option1].astype(float).dropna(), df[option2].astype(float).dropna())
            if(hip == "chi-square"):
                cross_tab = pd.crosstab(df[option1], df[option2], margins=True)
                ch_sq, pvalue, df, expected_ = stats.chi2_contingency(cross_tab)
                prediction = "chi-square: "+str(ch_sq)+" p-value: "+str(pvalue)+" df: "+str(df)
        else: st.error("Выберите разные колонки")
    st.success("The prediction is {}".format(prediction))