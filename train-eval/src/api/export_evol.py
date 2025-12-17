import csv
from google.oauth2 import service_account
from googleapiclient.discovery import build

def write_sheet_data_evol(i, name):
    # SERVICE_ACCOUNT_FILE = './src/api/dc-hyperparameter-search-95b4be9934ce.json'
    SERVICE_ACCOUNT_FILE = './src/api/preference-data-refinement-7f167538ce9d.json'
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    service = build('sheets', 'v4', credentials=credentials)
    sheet = service.spreadsheets()

    # SPREADSHEET_ID = '1vMDuJW2vcFHHRLAZkm55PdUhKO3i66ykQgY08I_0nsw'
    SPREADSHEET_ID = '1bcQ6GgWMI8igh-zeEkzQhcU1g6_9p3gVC0mpg_irXeM'
    append(i, name, 'V', 'pairwise-baseline', sheet, SPREADSHEET_ID)
    append(i, name, 'AC', 'single-1', sheet, SPREADSHEET_ID)
   
    print("✅ Evol-Instruct Data written to Google Sheet.")
    
def append(i, name, start, filename, sheet, SPREADSHEET_ID):
    RANGE_NAME = f"'{name}'!{start}{3*i+3}"

    csv_data = []
    with open(f'./eval/FastChat/fastchat/llm_judge/evol_instruct-gpt-4o-2024-11-20-{filename}.csv', mode='r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  
        for row in reader:
            print("raw row:", row)  # 테스트용
            numeric_row = []
            for val in row:
                try:
                    # 정수로 변환 가능한 경우 정수로
                    if '.' not in val:
                        numeric_row.append(int(val))
                    else:
                        numeric_row.append(float(val))  # 소수점이 있으면 float로
                except ValueError:
                    numeric_row.append(val)  # 숫자로 변환 불가하면 문자열로 유지
            csv_data.append(numeric_row)

    request = sheet.values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=RANGE_NAME,
        valueInputOption='RAW',
        body={'values': csv_data}
    )
    response = request.execute()
    return response