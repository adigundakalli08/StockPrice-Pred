import urllib.request
import json
import pandas as pd
import os

# URL to download the OpenAPI scrip master file
url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

def generate_scrip_master():

    try:
        with urllib.request.urlopen(url) as response:
            data = json.load(response)
            df = pd.DataFrame.from_dict(data, orient='columns')
            # Filter for NSE equity symbols
            df = df.loc[df['symbol'].str.endswith("-EQ")].sort_values(by=['symbol'])
            df.to_csv("ScripMaster.csv", index=False)
            print("Scrip master file generated successfully.")
    except Exception as e:
        print(f"Error generating Scrip Master: {e}")

def get_nse_scrip_token(symbol):
   
    if not os.path.exists("ScripMaster.csv"):
        generate_scrip_master()
    
    try:
        df = pd.read_csv("ScripMaster.csv")
        df = df.loc[df['name'] == symbol]
        if not df.empty:
            token = df['token'].iloc[0]
            return int(token)
        else:
            print(f"Symbol '{symbol}' not found in Scrip Master.")
            return None
    except Exception as e:
        print(f"Error retrieving scrip token: {e}")
        return None

