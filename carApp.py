import numpy as np
import streamlit as st
import joblib



#load our model and scaler
model = joblib.load("C:/Users/HP/Desktop/Car Price Predection/model.pkl")
scaler = joblib.load("C:/Users/HP/Desktop/Car Price Predection/scalar.pkl")


def car_price_predection(input_data):
    #changing the input into numpy array and reshaping
    input_changed = np.array(input_data).reshape(1,-1)

    #Standardize the model
    std_input = scaler.transform(input_changed)

    prediction = model.predict(std_input)

    return "Estimted car price : "+str(prediction[0])


def main():
    #creating the title

    st.title("Ford car price prediction App")


    #Getting the input form user
    year = st.text_input('Year')
    transmission = st.text_input('transmission')
    mileage = st.text_input('mileage')
    fuel_type = st.text_input('fuel Type')
    tax = st.text_input('Tax')
    mpg = st.text_input('mpg')
    enginesize = st.text_input('engine size')

    pred_price = ''

    #create a button
    if st.button('Check estimated price'):
        pred_price = car_price_predection([year,transmission,mileage,fuel_type,tax,mpg,enginesize])

    st.success(pred_price)
if __name__=='__main__':
    main()
