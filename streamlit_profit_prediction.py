import numpy as np
import pandas as pd
import streamlit as st
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectKBest, chi2

st.header("Model used:")
st.write("Linear Regression")
st.write("This profit prediction model utilizes linear regression to estimate profits based on key features like Segment, Region, Sales, Discount, Quantity, Ship Mode and Catgeory. By leveraging historical data that includes known profits and corresponding feature values, the model learns the underlying patterns and correlations. Through this learning process, the model develops a linear equation that best fits the relationship between the input features and the target variable (profit).")


st.title("Dataset")
set=pd.read_csv('Superstore.csv',encoding='windows-1254')
set
st.title('Modified Dataset')
df=pd.read_csv('modified superstore data.csv')
df

# model creation
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, train_size = 0.85, test_size = 0.15, random_state = 1)

from sklearn.linear_model import LinearRegression
X_train =df_train[['Sales','Quantity','Discount','First Class','Same Day',
             'Second Class','Standard Class','Consumer','Corporate','Home Office','Central','East','South',
             'West','Furniture','Office Supplies','Technology']] # independent variable

Y_train = df_train['Profit'] # dependent variable
lr = LinearRegression()
model = lr.fit(X_train, Y_train)

# --------------------------------------------------------------------------------------------------------

st.title("Profit Prediction")
with st.form(key="my_form1"):
    Category=st.radio("Category",["Furniture","Office Supplies","Technology"])
    ShipMode=st.radio("Ship Mode",["First Class","Same day","Second Class","Standard Class"])
    Segment=st.radio("Segment",["Consumer","Corporate","Home Office"])
    Region=st.radio("Region",["Central","East","South","West"])
    Sales=st.number_input("Enter sales value:")
    Quantity=st.number_input("Enter quantity:")
    Discount=st.number_input("Enter discount:")
    
    if Category=="Furniture":
        Furniture=1
        Office_supply=0
        Technology=0
    elif Category=="Office Supplies":
        Furniture=0
        Office_supply=1
        Technology=0
    else:
        Furniture=0
        Office_supply=0
        Technology=1

    if ShipMode=="First Class":
        First_class=1
        Same_day=0
        Second_class=0
        Standard_class=0
    if ShipMode=="Same day":
        First_class=0
        Same_day=1
        Second_class=0
        Standard_class=0
    if ShipMode=="Second Class":
        First_class=0
        Same_day=0
        Second_class=1
        Standard_class=0
    if ShipMode=="Standard Class":
        First_class=0
        Same_day=0
        Second_class=0
        Standard_class=1

    if Segment=="Consumer":
        Consumer=1
        Corporate=0
        Home_office=0
    if Segment=="Corporate":
        Consumer=0
        Corporate=1
        Home_office=0
    if Segment=="Home Office":
        Consumer=0
        Corporate=0
        Home_office=1

    if Region=="Central":
        Central=1
        East=0
        South=0
        West=0
    if Region=="East":
        Central=0
        East=1
        South=0
        West=0
    if Region=="South":
        Central=0
        East=0
        South=1
        West=0
    if Region=="Wast":
        Central=0
        East=0
        South=0
        West=1
    
    auto=pd.DataFrame({
        'Sales':[Sales],
        'Quantity':[Quantity],
        'Discount':[Discount],
        'First Class':[First_class],
        'Same Day':[Same_day],
        'Second Class':[Second_class],
        'Standard Class':[Standard_class],
        'Consumer':[Consumer],
        'Corporate':[Corporate],
        'Home Office':[Home_office],
        'Central':[Central],
        'East':[East],
        'South':[South],
        'West':[West],
        'Furniture':[Furniture],
        'Office Supplies':[Office_supply],
        'Technology':[Technology]
    })
    c=st.form_submit_button("calculate")

if c:
    auto
    output=model.predict(auto)
    st.write("Profit: ", output)

        

