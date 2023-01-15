import pickle 
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import category_encoders as ce
import streamlit as st 



#This function will predict the weekly sales for a department in a store.
def predict(x= [[1,1,False,42.31,2.572,211.096358,8.106,'A',151315,5,2010]]):
    df1 = pd.DataFrame(x,columns=['Store','Dept','IsHoliday','Temperature','Fuel_Price','CPI','Unemployment','Type','Size','Week','Year'])
    encoder=pickle.load(open('enc.pkl','rb'))
    encoded_dept = encoder.transform(df1['Dept'])
    df1 = df1.join(encoded_dept)
    df1 = pd.get_dummies(df1)
    model = pickle.load(open('model.pkl','rb'))
    columns=list(model.feature_names_in_)
    df1 = df1.reindex(columns = columns, fill_value=0)
    df1 = df1[columns]
    
    return model.predict(df1)



# This function will predict the weeklysales for a store. 
def predict2(x= [[1,False,42.31,2.572,211.096358,8.106,'A',151315,5,2010]]):
    df1 = pd.DataFrame(x,columns=['Store','IsHoliday','Temperature','Fuel_Price','CPI','Unemployment','Type','Size','Week','Year'])
    df1 = pd.get_dummies(df1)
    model = pickle.load(open('model2.pkl','rb'))
    columns=list(model.feature_names_in_)
    df1 = df1.reindex(columns = columns, fill_value=0)
    df1 = df1[columns]
    
    return model.predict(df1)

#This function takes input from the user and it will be redirected to either models depending on the inputs.
def main():
    st.title('Weekly Sales prediction')
    week=st.number_input("What is the week ?(1-52)")
    year=st.number_input('What is the Year that you are predicting in?')
    store=int(st.number_input("What is your store number(1-45)?"))
    st.write('If you want to predict the store sales please keep the next cell as zero.')
    dept=st.number_input("What is your Department number(0-99)?")
    isholiday=st.selectbox("Is there a holiday?",('True','False'))
    temp=st.number_input("What is the tremperature?(fahrenheit) ")
    fuel=st.number_input("What is the fuel price?(in $/gallon)")
    cpi=st.number_input("What is the CPI ?")
    unem=st.number_input("What is the Unemployment rate?")
    type=st.selectbox("What is the Type of your store?",('A','B','C'))
    size=st.number_input("What is the size?")
    
    weekly_sales=''
    if st.button('weekly_sales'):
        if dept==0:
            weekly_sales=predict2(x=[[store,isholiday,temp,fuel,cpi,unem,type,size,week,year]])
        else:
            weekly_sales=predict(x=[[store,dept,isholiday,temp,fuel,cpi,unem,type,size,week,year]])




    st.success(weekly_sales)


if __name__=='__main__':
    main()