# import the required packages
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
st.set_page_config(page_title='Voting Regressor Model', page_icon=':money_with_wings:')
@st.cache_data
def load_data():
    # First load the original airbnb listtings dataset
    data = pd.read_csv("listings_data.csv")
    return data
data = load_data()
st.sidebar.title("Exploring Airbnb Rental Price Estimation with Regression Model")

st.sidebar.markdown("This web app allows you to explore voting regressor model for the prediction of airbnb rental prices. You can filter the parameter neighbourhood group, room type, minimum nights,reviews per month and calculated host listings count. You can view the distribution of price (target) on a visualization in the 'Explore' tab and make predictions in the 'Predict' tab.")
room_type_dict = {
            'Entire home/apt': 3,
            'Private room': 2,
            'Shared room': 1
            }
ng_dict = {
            'Central Region': 1,
            'West Region': 2,
            'East Region': 3,
            'North-East Region':4,
            'North Region':5,
            }
neighbourhood_group = st.sidebar.selectbox("neighbourhood group", ['Central Region', 'West Region'   , 'East Region','North-East Region','North Region'], key='neighbourhood_group')
room_type = st.sidebar.selectbox("Room type", ['Entire home/apt','Private room' , 'Shared room'], key='room_type')
minimum_nights = st.sidebar.slider("minimum nights", 1.000000 ,365.000000  , (365.000000 ))
reviews_per_month = st.sidebar.slider("reviews per month", 0.00 , 3.060000   , ( 3.060000  ))
calculated_host_listings_count = st.sidebar.slider("calculated host listings count", 1.000000, 274.000000   , (274.000000 ))


tab1, tab2 = st.tabs(['Explore', 'Predict'])

with tab1:
    st.info('Feel free to adjust the parameters(left of the screen) and see how it changes the visualisations(right of the screen).', icon="ℹ️")
    st.title("Explore Airbnb Rental Prices with this Voting Regressor Model")
    filtered_data = data[data['neighbourhood_group'] == ng_dict[neighbourhood_group]]
    filtered_data = filtered_data[data['room_type'] == room_type_dict[room_type]]
    filtered_data=filtered_data[data['minimum_nights'] <= minimum_nights]
    filtered_data=filtered_data[data['reviews_per_month'] <= reviews_per_month]
    filtered_data=filtered_data[data['calculated_host_listings_count'] <= calculated_host_listings_count]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    st.subheader('Price distribution based on the provided parameter values')
    ax1.hist(filtered_data['price'], bins=50)
    ax1.set_xlabel('Price')
    ax1.set_ylabel('Price Probability Density function')
    ax2.boxplot(filtered_data['price'])
    st.pyplot(fig, use_container_width=True)
    st.subheader('Map of all rental apartments')
    st.map(filtered_data[["latitude" ,"longitude"]])
# Tab 2: Predict the Sepal Length using Linear Regression Model
with tab2:
    st.info('This is where the machine learning model tries to predict the prices of the airbnb rental apartments with the given parameters. Feel free to adjust the parameters and click the button "Predict Price".', icon="ℹ️")
    # Load the serialized trained model rf.pkl and scaler object scaler.pkl
    with open('regression.pkl', 'rb') as file:
        rf = pickle.load(file)

    # Define the app title and favicon
    st.title('Prediction of Airbnb rental price') 
    st.subheader('Predict')
    st.markdown("This tab allows you to make predictions on the price based on the input variables.")
    st.write('Specify the neighbourhood group, room type, minimum nights, reviews per month and calculated host listings count.')

    pred_neighbourhood_group = st.radio("neighbourhood group", ['Central Region', 'West Region'   , 'East Region','North-East Region','North Region'], index=0)
    pred_room_type = st.radio("room type", ['Entire home/apt','Private room' , 'Shared room'], index=0)
    pred_minimum_nights = st.number_input("minimum_nights",1.000000 ,365.000000  , 1.000000)    
    pred_reviews_per_month = st.number_input("reviews_per_month", 0.000000 , 3.060000  , 0.00)
    pred_calculated_host_listings_count = st.number_input("calculated_host_listings_count", 1.000000, 274.000000 , 1.000000)
    

    # Define a dictionary (n_mapping) that maps neighborhood to their corresponding integer values for each neighborhood group.
    
   
    # Create a function that takes neighbourhood_group as an argument and returns the corresponding integer value.
    # Create a price prediction button
    if st.button('Predict Price'):
        # Call the function with the selected room_type as an argument
        # Make the prediction
        if pred_neighbourhood_group=='Central Region':
            pred_neighbourhood=20
            pred_latitude=1.31125
            pred_longitude=103.86022 
        if pred_neighbourhood_group=='West Region':
            pred_neighbourhood=9
            pred_latitude=1.34678 
            pred_longitude=103.75972
        if pred_neighbourhood_group=='East Region':
            pred_neighbourhood=4
            pred_latitude=1.31857
            pred_longitude=103.91476
        if pred_neighbourhood_group=='North-East Region':
            pred_neighbourhood=19
            pred_latitude=1.38404
            pred_longitude=103.90128 
        if pred_neighbourhood_group=='North Region':
            pred_neighbourhood=1
            pred_latitude=1.35133
            pred_longitude=103.82090
        input_data = [[2413412 ,ng_dict[pred_neighbourhood_group],pred_neighbourhood,pred_latitude,pred_longitude,room_type_dict[pred_room_type],pred_minimum_nights,-1,pred_reviews_per_month,pred_calculated_host_listings_count]]
        input_df = pd.DataFrame(input_data, columns=[2413412  ,ng_dict[pred_neighbourhood_group],pred_neighbourhood,pred_latitude,pred_longitude,room_type_dict[pred_room_type],pred_minimum_nights,-1,pred_reviews_per_month,pred_calculated_host_listings_count])
        predicted_price = rf.predict(input_df)[0]
        st.write('room_type: ', pred_room_type)
        st.write('neighbourhood_group: ', pred_neighbourhood_group)
        st.write('minimum_nights: ', '{:,.2f}'.format(pred_minimum_nights))
        st.write('reviews_per_month: ', '{:,.2f}'.format(pred_reviews_per_month))
        st.write('calculated_host_listings_count: ', '{:,.2f}'.format(pred_calculated_host_listings_count))
        st.write('The predicted price is ', '${:,.2f}'.format(predicted_price))
