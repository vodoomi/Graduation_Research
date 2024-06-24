print("Hello, scraping.py!")

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

# ブラウザのオプションを格納する変数を生成
options = Options()
# ヘッドレスモードを有効にする
options.add_argument('--headless')
# ブラウザの起動を抑制する
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920,1080')
# エラーログを抑制する
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')


# ブラウザの起動
chromedriver = "C:/Users/abemi/chromedriver-win64/chromedriver.exe"
service = Service(executable_path=chromedriver)
driver = webdriver.Chrome(service=service)
url = "https://www.jalan.net/yad385995/kuchikomi/?contHideFlg=1&maxPrice=999999&rootCd=7701" \
      "&roomCrack=000000&screenId=UWW3701&idx=0&smlCd=141602&dateUndecided=1&minPrice=0" \
      "&yadNo=385995&callbackHistFlg=1&distCd=01"
driver.get(url)

# データを取得し、2次元配列に格納する
table = []
isContinue = True
while(isContinue):
  parentElems = driver.find_elements(By.CLASS_NAME, "jlnpc-kuchikomiCassette__rightArea")
  for parent_i in parentElems:
    # 評価値
    room = parent_i.find_elements(By.TAG_NAME,"dd")[0].text
    bath = parent_i.find_elements(By.TAG_NAME,"dd")[1].text
    breakfast = parent_i.find_elements(By.TAG_NAME,"dd")[2].text
    dinner = parent_i.find_elements(By.TAG_NAME,"dd")[3].text
    customerService = parent_i.find_elements(By.TAG_NAME,"dd")[4].text
    clean = parent_i.find_elements(By.TAG_NAME,"dd")[5].text
    # 表題
    head = parent_i.find_element(By.CLASS_NAME, 'jlnpc-kuchikomiCassette__lead').text
    # 改行を削除した口コミ本文
    body = parent_i.find_element(By.CLASS_NAME, 'jlnpc-kuchikomiCassette__postBody').text.replace("\n","")

    # 配列にデータを格納
    rowData = []
    rowData.append(room)
    rowData.append(bath)
    rowData.append(breakfast)
    rowData.append(dinner)
    rowData.append(customerService)
    rowData.append(clean)
    rowData.append(head)
    rowData.append(body)
    table.append(rowData)

  #　次へボタンがあったらボタンを押す、無ければフラグを変更してループを終了する
  try:
    driver.find_element(By.CSS_SELECTOR, "a.next")
    nextButton = driver.find_element(By.CSS_SELECTOR, "a.next")
    nextButton.click()
    time.sleep(2)
  except NoSuchElementException:
    isContinue = False

#　配列からDataFrameを作成し、csvで出力する
df = pd.DataFrame(table, columns=['部屋','風呂','料理（朝食）','料理（夕食）',
                                   '接客・サービス','清潔感','見出し','本文'])
df.index.name = 'index'
df.to_csv("../output/jaran/ホテル南風荘_口コミデータ.csv", encoding="utf-8")

#ブラウザの終了
driver.quit()