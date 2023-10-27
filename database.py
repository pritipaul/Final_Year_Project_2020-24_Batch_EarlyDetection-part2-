import os
import streamlit as st  
from deta import Deta
from dotenv import load_dotenv
import pandas as pd
load_dotenv(".env")

# DETA_KEY = "d0msktwripg_p6T7mVAf28nxSB1HnEEcrrXUpknPAaES"

DETA_KEY = os.getenv("DETA_KEY")
deta = Deta(DETA_KEY)

db = deta.Base("deta-base")
pdb = deta.Base("prediction")

def insert_period(data, expense):
    """Returns the report on a successful creation, otherwise raises an error"""
    period_data = {
        "data_categories": data,
        "expenses": expense
    }
    return db.put(period_data)

def insert_perdiction(email, result,features):
    """Returns the report on a successful creation, otherwise raises an error"""
    predict_data = {
        "edata": email,
        "prediction": result,
        "features":features
    }
    return pdb.put(predict_data)

def fetch_all_periods():
    """Returns a DataFrame of all periods for data visualization"""
    res = db.fetch()
    
    if not res.items:
        return None
    
    data_list = []
    expense_list = []
    
    for item in res.items:
        data_list.append(item['data_categories'])
        expense_list.append(item['expenses'])
    
    # Create a DataFrame for visualization
    data = {
        "Name": [data['name'] for data in data_list],
        "Email": [data['email'] for data in data_list],
        "Address": [data['address'] for data in data_list],
        "Phone Number": [data['phone'] for data in data_list],
        "Alternative Phone Number": [data['alternate_phone'] for data in data_list],
        "Gender": [expense['Gender'] for expense in expense_list],
        "Age": [expense['Age'] for expense in expense_list],
        "Years Of Education": [expense['Years Of Education'] for expense in expense_list],
        "Socioeconomic Status": [expense['Socioeconomic Status'] for expense in expense_list],
        "Mini Mental Stage Examination": [expense['Mini Mental Stage Examination'] for expense in expense_list],
        "Clinical Dementia Rating": [expense['Clinical Dementia Rating'] for expense in expense_list],
        "Estimated Total Intracranial Volume": [expense['Estimated Total Intracranial Volume'] for expense in expense_list],
        "Normalized Whole-Brain Volume": [expense['Normalized Whole-Brain Volume'] for expense in expense_list],
        "Auto Scaling Factor": [expense['Auto Scaling Factor'] for expense in expense_list]
    }
    
    df = pd.DataFrame(data)
    return df

def fetch_all_predictions():
    res = pdb.fetch()
    
    if not res.items:
        return None
    
    email_list = []
    prediction_list = []
    features_list = []
    
    for item in res.items:
        email_list.append(item['edata'])
        prediction_list.append(item['prediction'])
        features_list.append(item['features'])
    
    # Create a DataFrame for predictions and features
    data = {
        "Email": [email['Email'] for email in email_list],
        "Prediction": prediction_list,
        "Features": features_list,
    }
    
    df = pd.DataFrame(data)
    return df

