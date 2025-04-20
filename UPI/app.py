# import altair as alt
# import numpy as np
# import pandas as pd
# import streamlit as st
# import datetime
# from datetime import datetime as dt
# import time
# import base64
# import pickle 
# # import subprocess
# # subprocess.check_call(["pip", "install", "xgboost"])
# from xgboost import XGBClassifier

# """
# # Welcome to your own UPI Transaction Fraud Detector!

# You have the option of inspecting a single transaction by adjusting the parameters below OR you can even check 
# multiple transactions at once by uploading a .csv file in the specified format
# """

# pickle_file_path = "UPI Fraud Detection Final.pkl"
# # Load the saved XGBoost model from the pickle file
# loaded_model = pickle.load(open(pickle_file_path, 'rb'))

# tt = ["Bill Payment", "Investment", "Other", "Purchase", "Refund", "Subscription"]
# pg = ["Google Pay", "HDFC", "ICICI UPI", "IDFC UPI", "Other", "Paytm", "PhonePe", "Razor Pay"]
# ts = ['Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
# mc = ['Donations and Devotion', 'Financial services and Taxes', 'Home delivery', 'Investment', 'More Services', 'Other', 'Purchases', 'Travel bookings', 'Utilities']

# tran_date = st.date_input("Select the date of your transaction", datetime.date.today())
# if tran_date:
#     selected_date = dt.combine(tran_date, dt.min.time())
#     month = selected_date.month
#     year = selected_date.year

# tran_type = st.selectbox("Select transaction type", tt)
# pmt_gateway = st.selectbox("Select payment gateway", pg)
# tran_state=st.selectbox("Select transaction state",ts)
# merch_cat = st.selectbox("Select merchant category", mc)

# amt = st.number_input("Enter transaction amount",step=0.1)

# st.write("OR")

# df = pd.read_csv("sample.csv")
# st.write("CSV Format:", df)

# uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.write("Uploaded CSV:", df)

# button_clicked = st.button("Check transaction(s)")
# st.markdown(
#     """
#     <style>
#     .stButton>button {
#         position: fixed;
#         bottom: 40px;
#         left: 413px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
# if button_clicked:
#     tt_oh = []
#     for i in range(len(tt)):
#         tt_oh.append(0)
#     pg_oh = []
#     for i in range(len(pg)):
#         pg_oh.append(0)
#     ts_oh = []
#     for i in range(len(ts)):
#         ts_oh.append(0)
#     mc_oh = []
#     for i in range(len(mc)):
#         mc_oh.append(0)
#     if uploaded_file is not None:
#         with st.spinner("Checking transactions..."):
#             def download_csv():
#                 csv = df.to_csv(index=False,header=True)
#                 b64 = base64.b64encode(csv.encode()).decode()
#                 href = f'<a href="data:file/csv;base64,{b64}" download="output.csv">Download Output CSV</a>'
#                 return href
#             df[['Month', 'Year']] = df['Date'].str.split('-', expand=True)[[1, 2]]
#             df[['Month', 'Year']] = df[['Month', 'Year']].astype(int)
#             df.drop(columns=['Date'], inplace=True)
#             df = df.reindex(columns=['Amount', 'Year', 'Month','Transaction_Type','Payment_Gateway','Transaction_State','Merchant_Category'])
#             results = []
#             for index, row in df.iterrows():
#                 input = []
#                 input.append(row.values[0])
#                 input.append(row.values[1])
#                 input.append(row.values[2])
#                 tt_oh[tt.index(row.values[3])]=1
#                 pg_oh[pg.index(row.values[4])]=1
#                 ts_oh[ts.index(row.values[5])]=1
#                 mc_oh[mc.index(row.values[6])]=1
#                 input = input+tt_oh+pg_oh+ts_oh+mc_oh
#                 prediction = loaded_model.predict([input])[0]
#                 results.append(prediction)
#             df['fraud']=results
#             st.success("Checked transactions!")
#             st.markdown(download_csv(), unsafe_allow_html=True)
            
#     else:
#         with st.spinner("Checking transaction(s)..."):
#             tt_oh[tt.index(tran_type)]=1
#             pg_oh[pg.index(pmt_gateway)]=1
#             ts_oh[ts.index(tran_state)]=1
#             mc_oh[mc.index(merch_cat)]=1
#             input = []
#             input.append(amt)
#             input.append(year)
#             input.append(month)
#             input = input+tt_oh+pg_oh+ts_oh+mc_oh
#             inputs = [input]
#             result = loaded_model.predict(inputs)[0]
#             st.success("Checked transaction!")
#             if(result==0):
#                 st.write("Congratulations! Not a fraudulent transaction.")
#             else:
#                 st.write("Oh no! This transaction is fraudulent.")

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import datetime
from datetime import datetime as dt
import time
import base64
import pickle 
from xgboost import XGBClassifier

# Load Model
pickle_file_path = "UPI Fraud Detection Final.pkl"
loaded_model = pickle.load(open(pickle_file_path, 'rb'))

# Static lists
tt = ["Bill Payment", "Investment", "Other", "Purchase", "Refund", "Subscription"]
pg = ["Google Pay", "HDFC", "ICICI UPI", "IDFC UPI", "Other", "Paytm", "PhonePe", "Razor Pay"]
ts = ['Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 
      'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 
      'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
mc = ['Donations and Devotion', 'Financial services and Taxes', 'Home delivery', 'Investment', 'More Services', 'Other', 
      'Purchases', 'Travel bookings', 'Utilities']

# App Title
st.markdown("<h1 style='text-align: center; color: #00A6ED;'>🔍 UPI Transaction Fraud Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Check individual transactions or upload a CSV to detect fraud.</p>", unsafe_allow_html=True)

# Section: Single Transaction
st.markdown("### 👤 Single Transaction Check")

col1, col2 = st.columns(2)
with col1:
    tran_date = st.date_input("Transaction Date", datetime.date.today())
    if tran_date:
        selected_date = dt.combine(tran_date, dt.min.time())
        month = selected_date.month
        year = selected_date.year

    amt = st.number_input("💰 Transaction Amount", step=0.1)

with col2:
    tran_type = st.selectbox("📝 Transaction Type", tt)
    pmt_gateway = st.selectbox("🏦 Payment Gateway", pg)
    tran_state = st.selectbox("📍 Transaction State", ts)
    merch_cat = st.selectbox("🛒 Merchant Category", mc)

st.markdown("### 📄 OR Upload a CSV File")
st.markdown("**CSV Format Preview:**")
df = pd.read_csv("sample.csv")
st.dataframe(df, use_container_width=True)

uploaded_file = st.file_uploader("Upload your transaction CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("**Uploaded CSV Preview:**")
    st.dataframe(df, use_container_width=True)

# Submit Button
st.markdown("""<div style='margin-top: 30px; text-align: center;'>""", unsafe_allow_html=True)
button_clicked = st.button("🚀 Check Transaction(s)")
st.markdown("</div>", unsafe_allow_html=True)

# Style Enhancement
st.markdown("""
    <style>
    .stButton>button {
        background-color: #00A6ED;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #0076A3;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# On Click
if button_clicked:
    # One-hot initialization
    tt_oh = [0]*len(tt)
    pg_oh = [0]*len(pg)
    ts_oh = [0]*len(ts)
    mc_oh = [0]*len(mc)

    # Batch CSV Prediction
    if uploaded_file is not None:
        with st.spinner("🔍 Analyzing uploaded transactions..."):
            def download_csv():
                csv = df.to_csv(index=False, header=True)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="output.csv">📥 Download Output CSV</a>'
                return href

            df[['Month', 'Year']] = df['Date'].str.split('-', expand=True)[[1, 2]].astype(int)
            df.drop(columns=['Date'], inplace=True)
            df = df.reindex(columns=['Amount', 'Year', 'Month','Transaction_Type','Payment_Gateway','Transaction_State','Merchant_Category'])

            results = []
            for _, row in df.iterrows():
                input = [row['Amount'], row['Year'], row['Month']]
                # reset one-hot for each row
                tt_oh = [0]*len(tt)
                pg_oh = [0]*len(pg)
                ts_oh = [0]*len(ts)
                mc_oh = [0]*len(mc)
                tt_oh[tt.index(row['Transaction_Type'])] = 1
                pg_oh[pg.index(row['Payment_Gateway'])] = 1
                ts_oh[ts.index(row['Transaction_State'])] = 1
                mc_oh[mc.index(row['Merchant_Category'])] = 1
                input += tt_oh + pg_oh + ts_oh + mc_oh
                prediction = loaded_model.predict([input])[0]
                results.append(prediction)

            df['Fraudulent'] = ['❌ Fraud' if r else '✅ Safe' for r in results]
            st.success("Analysis complete!")
            st.dataframe(df, use_container_width=True)
            st.markdown(download_csv(), unsafe_allow_html=True)

    # Single Transaction Prediction
    else:
        with st.spinner("🔍 Checking your transaction..."):
            tt_oh[tt.index(tran_type)] = 1
            pg_oh[pg.index(pmt_gateway)] = 1
            ts_oh[ts.index(tran_state)] = 1
            mc_oh[mc.index(merch_cat)] = 1
            input = [amt, year, month] + tt_oh + pg_oh + ts_oh + mc_oh
            result = loaded_model.predict([input])[0]
            st.success("Check complete!")
            if result == 0:
                st.markdown("<h3 style='color: green;'>✅ This is a legitimate transaction.</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='color: red;'>⚠️ Warning! This transaction appears fraudulent.</h3>", unsafe_allow_html=True)
