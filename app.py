import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    return joblib.load("hotel_booking_pipeline.pkl")

@st.cache_resource
def create_encoders():
    """Create label encoders for categorical variables - MUST match training"""
    encoders = {}
    
    # Define categorical mappings (match exactly with training data)
    categorical_mappings = {
        'hotel': ['City Hotel', 'Resort Hotel'],
        'arrival_date_month': ['January', 'February', 'March', 'April', 'May', 'June',
                               'July', 'August', 'September', 'October', 'November', 'December'],
        'meal': ['BB', 'FB', 'HB', 'SC'],
        'country': ['PRT', 'GBR', 'FRA', 'DEU', 'ITA', 'ESP', 'Unknown'],  # Add more as needed
        'market_segment': ['Online TA', 'Offline TA/TO', 'Direct', 'Corporate', 'Groups'],
        'distribution_channel': ['TA/TO', 'Direct', 'Corporate', 'GDS'],
        'reserved_room_type': list('ABCDEFGH'),
        'deposit_type': ['No Deposit', 'Non Refund', 'Refundable'],
        'customer_type': ['Transient', 'Transient-Party', 'Contract', 'Group'],
        'arrival_season': ['Winter', 'Spring', 'Summer', 'Autumn']
    }
    
    for col, values in categorical_mappings.items():
        encoder = LabelEncoder()
        encoder.fit(values)
        encoders[col] = encoder
    
    return encoders

pipeline = load_pipeline()
encoders = create_encoders()

st.title("🏨 Hotel Booking Cancellation Predictor")

st.markdown(
    """
Predict whether a hotel booking is likely to be cancelled using a Machine Learning model.
"""
)

st.divider()

c1, c2, c3, c4 = st.columns(4)

c1.metric("🤖 Model", "Random Forest")

c2.metric("🎯 Task", "Classification")

c3.metric("📊 Features", "31")

c4.metric("🏨 Output", "Cancel / Stay")

st.divider()

st.sidebar.header("Booking Details")

hotel = st.sidebar.selectbox(
    "Hotel",
    ["City Hotel", "Resort Hotel"]
)

lead_time = st.sidebar.number_input(
    "Lead Time",
    0,
    800,
    30
)

arrival_date_year = st.sidebar.selectbox(
    "Arrival Year",
    [2015, 2016, 2017]
)

arrival_date_month = st.sidebar.selectbox(
    "Arrival Month",
    [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]
)

arrival_date_week_number = st.sidebar.slider(
    "Week Number",
    1,
    53,
    25
)

arrival_date_day_of_month = st.sidebar.slider(
    "Day",
    1,
    31,
    15
)

stays_in_weekend_nights = st.sidebar.slider(
    "Weekend Nights",
    0,
    10,
    1
)

stays_in_week_nights = st.sidebar.slider(
    "Week Nights",
    0,
    20,
    2
)

adults = st.sidebar.slider(
    "Adults",
    1,
    6,
    2
)

children = st.sidebar.slider(
    "Children",
    0,
    4,
    0
)

babies = st.sidebar.slider(
    "Babies",
    0,
    2,
    0
)

meal = st.sidebar.selectbox(
    "Meal",
    ["BB","HB","FB","SC"]
)

country = st.sidebar.text_input(
    "Country Code",
    "PRT"
)

market_segment = st.sidebar.selectbox(
    "Market Segment",
    [
        "Online TA",
        "Offline TA/TO",
        "Direct",
        "Corporate",
        "Groups"
    ]
)

distribution_channel = st.sidebar.selectbox(
    "Distribution Channel",
    [
        "TA/TO",
        "Direct",
        "Corporate",
        "GDS"
    ]
)

is_repeated_guest = st.sidebar.selectbox(
    "Repeated Guest",
    [0,1]
)

previous_cancellations = st.sidebar.number_input(
    "Previous Cancellations",
    0,
    20,
    0
)

previous_bookings_not_canceled = st.sidebar.number_input(
    "Previous Successful Bookings",
    0,
    100,
    0
)

reserved_room_type = st.sidebar.selectbox(
    "Reserved Room",
    list("ABCDEFGH")
)

deposit_type = st.sidebar.selectbox(
    "Deposit Type",
    [
        "No Deposit",
        "Non Refund",
        "Refundable"
    ]
)
agent = st.sidebar.number_input(
    "Agent ID",
    0,
    600,
    9
)

company = st.sidebar.number_input(
    "Company ID",
    0,
    600,
    0
)

days_in_waiting_list = st.sidebar.slider(
    "Waiting Days",
    0,
    400,
    0
)

customer_type = st.sidebar.selectbox(
    "Customer Type",
    [
        "Transient",
        "Transient-Party",
        "Contract",
        "Group"
    ]
)

adr = st.sidebar.number_input(
    "Average Daily Rate",
    0.0,
    600.0,
    100.0
)

required_car_parking_spaces = st.sidebar.slider(
    "Parking Spaces",
    0,
    5,
    0
)

total_of_special_requests = st.sidebar.slider(
    "Special Requests",
    0,
    5,
    1
)

total_guests = st.sidebar.slider(
    "Total Guests",
    1,
    10,
    2
)

total_nights = st.sidebar.slider(
    "Total Nights",
    1,
    30,
    2
)

is_family = st.sidebar.selectbox(
    "Family Booking",
    [0,1]
)

arrival_season = st.sidebar.selectbox(
    "Season",
    [
        "Winter",
        "Spring",
        "Summer",
        "Autumn"
    ]
)

# Create DataFrame with ENCODED categorical variables
input_df = pd.DataFrame({
    "hotel": [encoders['hotel'].transform([hotel])[0]],
    "lead_time": [int(lead_time)],
    "arrival_date_year": [int(arrival_date_year)],
    "arrival_date_month": [encoders['arrival_date_month'].transform([arrival_date_month])[0]],
    "arrival_date_week_number": [int(arrival_date_week_number)],
    "arrival_date_day_of_month": [int(arrival_date_day_of_month)],
    "stays_in_weekend_nights": [int(stays_in_weekend_nights)],
    "stays_in_week_nights": [int(stays_in_week_nights)],
    "adults": [int(adults)],
    "children": [float(children)],
    "babies": [int(babies)],
    "meal": [encoders['meal'].transform([meal])[0]],
    "country": [encoders['country'].transform([country])[0] if country in encoders['country'].classes_ else encoders['country'].transform(['Unknown'])[0]],
    "market_segment": [encoders['market_segment'].transform([market_segment])[0]],
    "distribution_channel": [encoders['distribution_channel'].transform([distribution_channel])[0]],
    "is_repeated_guest": [int(is_repeated_guest)],
    "previous_cancellations": [int(previous_cancellations)],
    "previous_bookings_not_canceled": [int(previous_bookings_not_canceled)],
    "reserved_room_type": [encoders['reserved_room_type'].transform([reserved_room_type])[0]],
    "deposit_type": [encoders['deposit_type'].transform([deposit_type])[0]],
    "agent": [float(agent)],
    "company": [float(company)],
    "days_in_waiting_list": [int(days_in_waiting_list)],
    "customer_type": [encoders['customer_type'].transform([customer_type])[0]],
    "adr": [float(adr)],
    "required_car_parking_spaces": [int(required_car_parking_spaces)],
    "total_of_special_requests": [int(total_of_special_requests)],
    "total_guests": [int(total_guests)],
    "total_nights": [int(total_nights)],
    "is_family": [int(is_family)],
    "arrival_season": [encoders['arrival_season'].transform([arrival_season])[0]]
})

predict = st.button("Predict Booking Status", use_container_width=True)

if predict:
    try:
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0]

        cancel_prob = probability[1] * 100
        not_cancel_prob = probability[0] * 100
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Cancellation Probability",
                f"{cancel_prob:.2f}%"
            )
            st.progress(cancel_prob / 100)

        with col2:
            st.metric(
                "Booking Confirmation Probability",
                f"{not_cancel_prob:.2f}%"
            )
            st.progress(not_cancel_prob / 100)

        st.divider()

        if prediction == 1:
            st.error("❌ Booking is likely to be CANCELLED")
            st.subheader("Recommendation")
            st.write("""
- Contact the customer before arrival.
- Request advance payment.
- Send booking reminders.
- Consider flexible pricing strategies.
            """)
        else:
            st.success("✅ Booking is likely to be CONFIRMED")
            st.subheader("Recommendation")
            st.write("""
- Booking appears stable.
- Standard confirmation process is sufficient.
- Maintain regular customer communication.
            """)

        st.subheader("Input Summary")
        summary_df = pd.DataFrame({
            "Field": ["Hotel", "Lead Time", "Season", "Total Guests", "ADR"],
            "Value": [hotel, lead_time, arrival_season, total_guests, f"${adr:.2f}"]
        })
        st.dataframe(summary_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        st.info("Please check your input values and try again.")

st.divider()

st.markdown("""
### 📊 About This Project
This application predicts whether a hotel booking is likely to be cancelled using a **Random Forest Machine Learning model** trained on historical hotel booking data.

**Project Features**
- Hotel Booking Cancellation Prediction
- Random Forest Classifier
- Data Preprocessing Pipeline
- Interactive Streamlit Interface
- Probability-based Prediction
""")

st.divider()

st.caption(
    "Developed by Sarvan Bishnoi | Hotel Booking Analytics & Cancellation Prediction"
)
