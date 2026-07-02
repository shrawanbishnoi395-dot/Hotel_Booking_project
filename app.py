import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    return joblib.load("hotel_booking_pipeline.pkl")

pipeline = load_pipeline()

st.title("🏨 Hotel Booking Cancellation Predictor")
st.markdown("Predict whether a hotel booking is likely to be cancelled.")

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

# FIXED: Ensure all data types match exactly what pipeline expects
input_df = pd.DataFrame({
    "hotel": [hotel],  # Keep as string - categorical
    "lead_time": [int(lead_time)],
    "arrival_date_year": [int(arrival_date_year)],
    "arrival_date_month": [arrival_date_month],  # Keep as string - categorical
    "arrival_date_week_number": [int(arrival_date_week_number)],
    "arrival_date_day_of_month": [int(arrival_date_day_of_month)],
    "stays_in_weekend_nights": [int(stays_in_weekend_nights)],
    "stays_in_week_nights": [int(stays_in_week_nights)],
    "adults": [int(adults)],
    "children": [float(children)],  # Float for median imputation
    "babies": [int(babies)],
    "meal": [meal],  # Keep as string - categorical
    "country": [country],  # Keep as string - categorical
    "market_segment": [market_segment],  # Keep as string - categorical
    "distribution_channel": [distribution_channel],  # Keep as string - categorical
    "is_repeated_guest": [int(is_repeated_guest)],
    "previous_cancellations": [int(previous_cancellations)],
    "previous_bookings_not_canceled": [int(previous_bookings_not_canceled)],
    "reserved_room_type": [reserved_room_type],  # Keep as string - categorical
    "deposit_type": [deposit_type],  # Keep as string - categorical
    "agent": [float(agent)],  # Float for median imputation on missing values
    "company": [float(company)],  # Float for median imputation on missing values
    "days_in_waiting_list": [int(days_in_waiting_list)],
    "customer_type": [customer_type],  # Keep as string - categorical
    "adr": [float(adr)],
    "required_car_parking_spaces": [int(required_car_parking_spaces)],
    "total_of_special_requests": [int(total_of_special_requests)],
    "total_guests": [int(total_guests)],
    "total_nights": [int(total_nights)],
    "is_family": [int(is_family)],
    "arrival_season": [arrival_season]  # Keep as string - categorical
})

# Fill NaN values in numeric columns before prediction
numeric_cols = input_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if input_df[col].isna().any():
        input_df[col] = input_df[col].fillna(0)

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

        st.subheader("Booking Summary")
        st.dataframe(input_df, use_container_width=True)
        
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
