# coding: utf-8
import pandas as pd

if __name__ == "__main__":
	jp = pd.read_table("wo_empty_line_jp.txt", names =('jp_article',))
	en = pd.read_table("wo_empty_line_en.txt", names =('en_article',))
	en.to_csv("en_news.csv")
	jp.to_csv("jp_news.csv")