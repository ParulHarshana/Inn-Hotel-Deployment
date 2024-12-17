import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open('final_model_xgb.pkl','rb') as file:
    model=pickle.load(file)

with open('transformer.pkl','rb' ) as file:
    pt=pickle.load(file)

def prediction(input_list):
    input_list=np.array(input_list,dtype=object)
    
    pred=model.predict_proba([input_list])[:,1][0]
    
    if pred>0.5:
        return f'This booking is more likely to get cancelled: Chances {round((pred),2)}'
    else:
        return f'This booking is less likely to get cancelled: Chances {round((pred),2)}'

def main():
    st.title('INN HOTEL GROUP')
    lt=st.test_input('Enter the lead time.')
    mst=(lambda x: 1 if x=='Online'else 0)(st.st.selectbox('Enter the type of booking',['Online','Offline']))
    spcl=st.selectbox('Select the no of special request made',[1,2,3,4,5])
    price=st.text_input('Enter the price offered for the room')
    adults=st.selectbox('Select the no of adults in the booking',[0,1,2,3,4,])
    weekend=st.text_input('Enter the weekend nights in the booking')
    weeknights=st.text_input('Enter the week nights in the booking')
    park=(lambda x: 1 if x=='Yes' else 0)(st.selectbox('Is parking included in the booking',['Yes','No']))
    month=st.slider('What will be the month of arrival',min_value=1,max_value=12,step=1)
    day=st.slider('What will be the day of arrival',min_value=1,max_value=31,step=1)
    weekday=(lambda x: 0 if x=='Mon' else 1 if x=='Tue' else 2 if x=='Wed' else 3 if x=='Thur' else 4 if x=='Fri' else 5 if x=='Sat' else 6)
    (st.selectbox('What is the week day of arrival',['Mon','Tue','Wed','Thur','Fri','sat','sun']))

    trans_data=pt.transform([[lt,price]])
    lt_t=trans_data[0][0]
    price_t=trans_data[0][1]

    input_list=[lt_t,mst,spcl,price_t,adults,weekend,park,weeknights,month,day,weekday]

    if st.button('Predict'):
        response=predict(input_list)
        st.success(response)

if __name__=='__main__':
    main()
    

