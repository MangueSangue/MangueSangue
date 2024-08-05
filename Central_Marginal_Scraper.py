#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
import pandas as pd
from datetime import datetime

def fetch_and_save_data():
    # URL to fetch data
    url_d = 'https://apps.oc.org.do/wsOCWebsiteChart/Service.asmx/GetCentralMarginalPonderadaJSon?Fecha=07/20/2024'
    response = requests.get(url_d)
    response.raise_for_status()
    data = response.json()

    # Extract the relevant part of the data
    nested_data = data['GetCentralMarginalPonderada']
    df = pd.json_normalize(nested_data)

    # Prepare the filename with the current date
    today_date = datetime.now().strftime('%Y-%m-%d')
    filename = f'daily_costos_marginales_{today_date}.xlsx'

    # Append to the existing Excel file
    try:
        with pd.ExcelWriter(filename, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            df.to_excel(writer, sheet_name=today_date, index=False)
    except FileNotFoundError:
        df.to_excel(filename, sheet_name=today_date, index=False)

    print(f"Data saved to {filename}")

if __name__ == "__main__":
    fetch_and_save_data()


