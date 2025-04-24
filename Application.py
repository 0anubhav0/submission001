import streamlit as st
import joblib
import pandas as pd

model = joblib.load("best_model.pkl")

risk = ["good risk", "bad risk"]

def classify(props):
    args = {
        "Age": props["Age"],
        "Credit amount": props["Credit amount"],
        "Duration": props["Duration"],

        'Sex_male': props["Sex"] == "male",

        'Job_1': False, 'Job_2': False, 'Job_3': False,
        'Housing_own': False, 'Housing_rent': False, 'Saving accounts_moderate': False,
        'Saving accounts_quite rich': False, 'Saving accounts_rich': False,
        'Checking account_moderate': False, 'Checking account_rich': False,
        'Purpose_car': False, 'Purpose_domestic appliances': False, 'Purpose_education': False, 'Purpose_furniture/equipment': False, 'Purpose_radio/TV': False, 'Purpose_repairs': False, 'Purpose_vacation/others': False
    }

    if args.__contains__(props["Job"]):
        args[props["Job"]] = True

    if args.__contains__(props["Housing"]):
        args[props["Housing"]] = True

    if args.__contains__(props["Saving accounts"]):
        args[props["Saving accounts"]] = True

    if args.__contains__(props["Checking account"]):
        args[props["Checking account"]] = True

    if args.__contains__(props["Purpose"]):
        args[props["Purpose"]] = True

    print(args)

    return "prediction:  " + risk[int(model.predict(pd.DataFrame([args])))]

st.text("Credit Risk Prediction Form")

age = st.number_input("Age", min_value=18)
sex = st.selectbox("Sex", options=["male", "female"])
credit_amount = st.number_input("Credit Amount (in DM)", min_value=0)
duration = st.number_input("Duration (in months)", min_value=1)

housing = "Housing_" + st.selectbox("Housing", options=["own", "rent", "free"])
saving_accounts = "Saving accounts_" + st.selectbox("Saving Accounts", options=["little", "moderate", "quite rich", "rich"])
checking_accounts = "Checking account_" + st.selectbox("Checking Accounts", options=["little", "moderate", "rich"])
purpose = "Purpose_" + st.selectbox("Purpose", options=["car", "furniture/equipment", "radio/TV", "domestic appliances", "repairs", "education", "business", "vacation/others"])

jobMap = {
    "unskilled and non-resident": "Job_0",
    "unskilled and resident": "Job_1",
    "skilled": "Job_2",
    "highly skilled": "Job_3"
}

job = jobMap[st.selectbox("Job", options=jobMap.keys())]

if st.button("Submit"):
    st.text(classify({
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving_accounts,
        "Checking account": checking_accounts,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose
    }))
