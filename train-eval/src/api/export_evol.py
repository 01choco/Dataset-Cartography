import csv
from google.oauth2 import service_account
from googleapiclient.discovery import build

def write_sheet_data_evol(i, dataset):
    SERVICE_ACCOUNT_FILE = './src/api/dc-hyperparameter-search-95b4be9934ce.json'
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    service = build('sheets', 'v4', credentials=credentials)
    sheet = service.spreadsheets()

    SPREADSHEET_ID = '1vMDuJW2vcFHHRLAZkm55PdUhKO3i66ykQgY08I_0nsw'

    append(i, dataset, 'S', 'pairwise-baseline', sheet, SPREADSHEET_ID)
    append(i, dataset, 'Z', 'single-1', sheet, SPREADSHEET_ID)
   
    print("✅ Evol-Instruct Data written to Google Sheet.")
    
def append(i, dataset, start, filename, sheet, SPREADSHEET_ID):
    dataset = list(dataset)[0]
    RANGE_NAME = f"'{dataset}'!{start}{3*i+3}"

    csv_data = []
    with open(f'./eval/FastChat/fastchat/llm_judge/evol_instruct-gpt-4o-2024-11-20-{filename}.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  
        for row in reader:
            csv_data.append(row)

    request = sheet.values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=RANGE_NAME,
        valueInputOption='RAW',
        body={'values': csv_data}
    )
    response = request.execute()
    return response