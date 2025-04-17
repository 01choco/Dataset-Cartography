from google.oauth2 import service_account
from googleapiclient.discovery import build

def get_sheet_data(model, typ):
    SERVICE_ACCOUNT_FILE = './src/api/dc-hyperparameter-search-95b4be9934ce.json'
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    service = build('sheets', 'v4', credentials=credentials)

    SPREADSHEET_ID = '1vMDuJW2vcFHHRLAZkm55PdUhKO3i66ykQgY08I_0nsw'
    model = list(model)[0]
    typ = list(typ)[0]
    RANGE_NAME = f"'{model}-{typ}'!A3:E"


    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
    values = result.get('values', [])

    values = [row for row in values if len(row) > 0]
    print(values)
    return values