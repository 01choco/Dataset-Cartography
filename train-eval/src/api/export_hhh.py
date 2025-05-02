import csv
from google.oauth2 import service_account
from googleapiclient.discovery import build

def write_sheet_data_hhh(i, dataset, models):
    SERVICE_ACCOUNT_FILE = './src/api/dc-hyperparameter-search-95b4be9934ce.json'
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    service = build('sheets', 'v4', credentials=credentials)
    sheet = service.spreadsheets()

    SPREADSHEET_ID = '1vMDuJW2vcFHHRLAZkm55PdUhKO3i66ykQgY08I_0nsw'

    append(i, dataset, 'AQ', models, sheet, SPREADSHEET_ID)
   
    print("✅ HHH Data written to Google Sheet.")
    
def append(i, dataset, start, models, sheet, SPREADSHEET_ID):

    RANGE_NAME = f"'{dataset}'!{start}{3*i+3}"

    csv_data = []
    print("models:", models)  # 테스트용 출력
    for model in models:

        with open(f'./eval/instruct-eval/results/results_hhh_{model}.csv', mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                print("raw row:", row)  # 테스트용
                numeric_row = []
                for val in row:
                    try:
                        numeric_row.append(float(val))  # 숫자로 변환 시도
                    except ValueError:
                        numeric_row.append(val)  # 숫자가 아니면 그냥 문자열로 유지
                csv_data.append(numeric_row)
        request = sheet.values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=RANGE_NAME,
            valueInputOption='RAW',
            body={'values': csv_data}
        )
        response = request.execute()
    return response