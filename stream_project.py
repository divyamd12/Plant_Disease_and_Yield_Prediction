
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt



st.set_page_config(
    page_title="Mini-Project- FarmLit",
    page_icon="🚀",
    layout="centered"  # or "centered"
)
st.title("🚀 Farmer's App- FarmLit 🚀")
st.image("download1.jpeg", use_column_width=True)

st.write("""
### MINI PROJECT

Problem Statement- To develop a portal for farmers to understand and improvise yield production and detect crop diseases

#### YIELD PREDICTION
###### Enter the suitable data as per your needs and check the yield prediction done 
""")

model_lr = pickle.load(open('model_lr_crop','rb'))
model_rfr = pickle.load(open('model_rfr_crop','rb'))

#model_crop_disease = load_model('model1.keras')


# Sample list of options
options_state = ['Andaman and Nicobar Islands', 'Andhra Pradesh',
       'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh',
       'Chhattisgarh', 'Dadra and Nagar Haveli', 'Goa', 'Gujarat',
       'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir ', 'Jharkhand',
       'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
       'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry',
       'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana ',
       'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']

# Create a dropdown menu
selected_state = st.selectbox('Select your state:', options_state)

# Sample list of options
options_season = ['Kharif     ', 'Whole Year ', 'Autumn     ', 'Rabi       ',
       'Summer     ', 'Winter     ']

# Create a dropdown menu
selected_season = st.selectbox('Select your season of the crop:', options_season)

# Sample list of options
options_crop = ['Arecanut', 'Other Kharif pulses', 'Rice', 'Banana', 'Cashewnut',
       'Coconut ', 'Dry ginger', 'Sugarcane', 'Sweet potato', 'Tapioca',
       'Black pepper', 'Dry chillies', 'other oilseeds', 'Turmeric',
       'Maize', 'Moong(Green Gram)', 'Urad', 'Arhar/Tur', 'Groundnut',
       'Sunflower', 'Bajra', 'Castor seed', 'Cotton(lint)', 'Horse-gram',
       'Jowar', 'Korra', 'Ragi', 'Tobacco', 'Gram', 'Wheat', 'Masoor',
       'Sesamum', 'Linseed', 'Safflower', 'Onion', 'other misc. pulses',
       'Samai', 'Small millets', 'Coriander', 'Potato',
       'Other  Rabi pulses', 'Soyabean', 'Beans & Mutter(Vegetable)',
       'Bhindi', 'Brinjal', 'Citrus Fruit', 'Cucumber', 'Grapes', 'Mango',
       'Orange', 'other fibres', 'Other Fresh Fruits', 'Other Vegetables',
       'Papaya', 'Pome Fruit', 'Tomato', 'Mesta', 'Cowpea(Lobia)',
       'Lemon', 'Pome Granet', 'Sapota', 'Cabbage', 'Rapeseed &Mustard',
       'Peas  (vegetable)', 'Niger seed', 'Bottle Gourd', 'Varagu',
       'Garlic', 'Ginger', 'Oilseeds total', 'Pulses total', 'Jute',
       'Peas & beans (Pulses)', 'Blackgram', 'Paddy', 'Pineapple',
       'Barley', 'Sannhamp', 'Khesari', 'Guar seed', 'Moth',
       'Other Cereals & Millets', 'Cond-spcs other', 'Turnip', 'Carrot',
       'Redish', 'Arcanut (Processed)', 'Atcanut (Raw)',
       'Cashewnut Processed', 'Cashewnut Raw', 'Cardamom', 'Rubber',
       'Bitter Gourd', 'Drum Stick', 'Jack Fruit', 'Snak Guard', 'Tea',
       'Coffee', 'Cauliflower', 'Other Citrus Fruit', 'Water Melon',
       'Total foodgrain', 'Kapas', 'Colocosia', 'Lentil', 'Bean',
       'Jobster', 'Perilla', 'Rajmash Kholar', 'Ricebean (nagadal)',
       'Ash Gourd', 'Beet Root', 'Lab-Lab', 'Ribed Guard', 'Yam',
       'Pump Kin', 'Apple', 'Peach', 'Pear', 'Plums', 'Litchi', 'Ber',
       'Other Dry Fruit', 'Jute & mesta']

# Create a dropdown menu
selected_crop = st.selectbox('Select your crop:', options_crop)

area = st.slider("Select the area of your farm land in meter square", 100, 13000, help="Slide me!")


#preparing input data

df = pd.read_csv("data_get.csv")


n1 = np.zeros((1, 165))


columns_input = np.loadtxt("columns_input.csv", delimiter=',', dtype=str).squeeze()
columns_input= np.array(columns_input)


df_input = pd.DataFrame(data=n1, columns=columns_input)


input_state= "State_Name_"+selected_state
input_crop= "Crop_"+selected_crop
input_season = "Season_"+selected_season

condition = (((df[input_crop] == 1) | (df[input_season] == 1) | (df[input_state] == 1) ) & ( (area-3000 < df['Area']) & (df['Area'] < area+3000)))

df_mean_input = df.loc[condition]
mean_production = df_mean_input.percent_of_production.mean()
# st.write(mean_production)


production_mean = df_mean_input.percent_of_production.mean()


df_input.loc[0, 'Area'] = area
df_input.loc[0, input_state] = 1
df_input.loc[0, input_crop] = 1
df_input.loc[0, input_season] = 1
if not np.isnan(mean_production):
       df_input.loc[0,'percent_of_production'] = mean_production
else:
       df_input.loc[0,'percent_of_production'] = df.percent_of_production.min()




y_pred_lr = model_lr.predict(df_input)
round_lr = round(y_pred_lr[0][0], 2)
y_pred_rfr = model_rfr.predict(df_input)
round_rfr = round(y_pred_rfr[0], 2)

st.write(f"The predicted yield of your farm by Linear Regression model is -- {round_lr} Kilogram")
st.write(f"The predicted yield of your farm by Random Forest model is -- {round_rfr} Kilogram")


# st.write("""
# ##### ✨
# ##### ✨
# ##### ✨

# #### DISEASE PREDICTION
# ###### Upload Image of your plant and check for healhy/unhealthy status
# """)

# from PIL import Image
# import io

# # Use st.file_uploader to upload an image
# uploaded_image = st.file_uploader("Choose an image", type=["jpg","jpeg"])

# # Check if an image is uploaded
# if uploaded_image is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_image)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Process the image (you can add your image processing logic here)

#     # Add a download button
#     if st.button("Download Processed Image"):
#         # Save the processed image to a BytesIO buffer
#         buffer = io.BytesIO()
#         image.save(buffer, format="JPEG")
#         buffer.seek(0)

#         # Provide a download link
#         st.download_button(
#             label="Download",
#             data=buffer,
#             file_name="processed_image.jpg",
#             key="download_button",
#        )

















# # Load and preprocess the input image

# image_path = 'C:\\Users\\Dell\\Downloads\\processed_image.jpg'
# # Load and preprocess the image
# img = image.load_img(image_path, target_size=(150, 150))
# img = image.img_to_array(img)
# img = np.expand_dims(img, axis=0)
# img = img / 255.0  

# # Make predictions
# predictions = model.predict(img)
# predicted_class_index = np.argmax(predictions)
# predicted_class = class_labels[predicted_class_index]

# # Make predictions
# predictions = model_crop_disease.predict(img_array)
