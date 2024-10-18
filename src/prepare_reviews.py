import os
import zipfile
import gc

import polars as pl
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
unzip_password = os.getenv("UNZIP_PASSWORD")

# ディレクトリパスを指定
directory = '../data/2024年度_楽天トラベルデータ/01_Travel_Review'
output_csv = '../output/rakuten/all_reviews.csv'

# TSVファイルを格納するリスト
all_dataframes = []

# ディレクトリ内のすべてのzipファイルを処理
for filename in tqdm(os.listdir(directory)):
    if filename.endswith('.zip'):
        # zipファイルのパス
        zip_path = os.path.join(directory, filename)

        # zipファイル名から解凍先のフォルダ名を推定（ファイル名と同じ名前に）
        zip_folder_name = os.path.splitext(filename)[0]
        tsv_path = os.path.join(directory, zip_folder_name)

        # 解凍先のディレクトリが存在するかを確認
        if not os.path.exists(tsv_path):
            # 解凍ディレクトリが存在しない場合、zipファイルを解凍
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tsv_path, pwd=unzip_password.encode('utf-8'))
        else:
            pass

        # TSVファイルを読み込み
        col_names = [
            'poster_id',        # 投稿者ID：「user_1」などマスクしたユーザ名
            'post_date',        # 投稿日時：評価登録年月日（フォーマット「yyyy-mm-dd HH:MM:SS」）
            'facility_id',      # 施設ID：施設名は(2)の 02_Travel_HotelMaster.tsv を参照のこと
            'plan_id',          # プランID：プランIDの数字
            'plan_title',       # プランタイトル：プランタイトル文字列
            'room_type',        # 部屋種類：「s」「s1」などの文字列
            'room_name',        # 部屋名前：部屋の名前の文字列
            'purpose',          # 目的：「レジャー」、「ビジネス」などの文字列
            'companion',        # 同伴者：「家族」、「一人」などの文字列
            'rating_location',  # 評価１（立地）：0-5の6段階評価
            'rating_room',      # 評価２（部屋）：0-5の6段階評価
            'rating_food',      # 評価３（食事）：0-5の6段階評価
            'rating_bath',      # 評価４（風呂）：0-5の6段階評価
            'rating_service',   # 評価５（サービス）：0-5の6段階評価
            'rating_facilities',# 評価６（設備）：0-5の6段階評価
            'rating_overall',   # 評価７（総合）：0-5の6段階評価
            'review',      # ユーザ投稿本文：投稿本文の文字列
            'facility_reply'    # 施設回答本文：投稿に対する施設側からのコメント
        ]
        df = pl.read_csv(tsv_path, separator='\t')
        df.columns = col_names

        # データフレームをリストに追加
        all_dataframes.append(df.select(["post_date", "facility_id", "review"]))

        del df
        gc.collect()

# すべてのデータフレームを結合
combined_df = pl.concat(all_dataframes)

# CSVとして出力
combined_df.write_csv(output_csv)

print("Done!")