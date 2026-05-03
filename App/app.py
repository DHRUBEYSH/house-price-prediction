import joblib
import streamlit as st
import pandas as pd

# Load
model = joblib.load("Models/model.pkl")
location_encoder = joblib.load("Models/location_encoder.pkl")
city_encoder = joblib.load("Models/city_encoder.pkl")
features = joblib.load("Models/features.pkl")

st.title("🏠 House Price Prediction")

# ---------- BASIC DETAILS ----------
st.subheader("Basic Details")

area = st.slider("Area (sq ft)", 500, 5000, 1200)
bedrooms = st.slider("No. of Bedrooms", 1, 5, 2)

location = st.selectbox("Location", location_encoder.classes_)
city = st.selectbox("City", city_encoder.classes_)

latitude = st.number_input("Latitude", value=12.97)
longitude = st.number_input("Longitude", value=77.59)

# ---------- AMENITIES ----------
st.subheader("Amenities")

def checkbox(label):
    return 1 if st.checkbox(label) else 0

resale = checkbox("Resale")
gymnasium = checkbox("Gymnasium")
swimmingpool = checkbox("Swimming Pool")
landscapedgardens = checkbox("Landscaped Gardens")
joggingtrack = checkbox("Jogging Track")
rainwaterharvesting = checkbox("Rain Water Harvesting")
indoorgames = checkbox("Indoor Games")
shoppingmall = checkbox("Shopping Mall")
intercom = checkbox("Intercom")
sportsfacility = checkbox("Sports Facility")
atm = checkbox("ATM")
clubhouse = checkbox("Club House")
school = checkbox("School")
security = checkbox("24X7 Security")
powerbackup = checkbox("Power Backup")
carparking = checkbox("Car Parking")
staffquarter = checkbox("Staff Quarter")
cafeteria = checkbox("Cafeteria")
multipurpose = checkbox("Multipurpose Room")
hospital = checkbox("Hospital")
gas = checkbox("Gas Connection")
ac = checkbox("AC")
wifi = checkbox("WiFi")
children = checkbox("Children Play Area")
lift = checkbox("Lift Available")
vaastu = checkbox("Vaastu Compliant")
golf = checkbox("Golf Course")
tv = checkbox("TV")
dining = checkbox("Dining Table")
sofa = checkbox("Sofa")
wardrobe = checkbox("Wardrobe")
fridge = checkbox("Refrigerator")

# ---------- PREDICT ----------
if st.button("Predict Price"):

    new_data = pd.DataFrame({
        'Area': [area],
        'Location': [location],
        'No. of Bedrooms': [bedrooms],
        'Resale': [resale],
        'MaintenanceStaff': [0],  # default
        'Gymnasium': [gymnasium],
        'SwimmingPool': [swimmingpool],
        'LandscapedGardens': [landscapedgardens],
        'JoggingTrack': [joggingtrack],
        'RainWaterHarvesting': [rainwaterharvesting],
        'IndoorGames': [indoorgames],
        'ShoppingMall': [shoppingmall],
        'Intercom': [intercom],
        'SportsFacility': [sportsfacility],
        'ATM': [atm],
        'ClubHouse': [clubhouse],
        'School': [school],
        '24X7Security': [security],
        'PowerBackup': [powerbackup],
        'CarParking': [carparking],
        'StaffQuarter': [staffquarter],
        'Cafeteria': [cafeteria],
        'MultipurposeRoom': [multipurpose],
        'Hospital': [hospital],
        'Gasconnection': [gas],
        'AC': [ac],
        'Wifi': [wifi],
        "Children'splayarea": [children],
        'LiftAvailable': [lift],
        'VaastuCompliant': [vaastu],
        'GolfCourse': [golf],
        'TV': [tv],
        'DiningTable': [dining],
        'Sofa': [sofa],
        'Wardrobe': [wardrobe],
        'Refrigerator': [fridge],
        'City': [city],
        'Latitude': [latitude],
        'Longitude': [longitude]
    })

    # Encoding
    new_data['Location'] = location_encoder.transform(new_data['Location'])
    new_data['City'] = city_encoder.transform(new_data['City'])

    # Match features
    for col in features:
        if col not in new_data.columns:
            new_data[col] = 0

    new_data = new_data[features]

    prediction = model.predict(new_data)

    st.success(f"🏷️ Estimated Price: ₹ {prediction[0]:,.2f}")