# 导入库
import pandas as pd
# 传入要抓取的url
html = "https://www.cnopendata.com/data/m/recently-updated-data/china-customs-statistics/import&exportgoods-gross-value-country.html"
constitution_html = "https://www.cnopendata.com/data/m/recently-updated-data/china-customs-statistics/import&exportgoods-constitution.html"
index_html = "./（1）进出口商品总值表(人民币值) A_年度表.html"
# 0表示选中网页中的第一个Table
# df1 = pd.read_html(html)[0]
# df1 = pd.read_html(html)[1]
# df2 = pd.read_html(constitution_html)[1]
df3 = pd.read_html(index_html)[2]
# 打印预览
# print(df1)
print(df3)
# df1m
# # 导出到CSV
# df1.to_csv('./进出口商品国别（地区）总值表.csv', index=0, encoding="utf-8")
# df2.to_csv('./进出口商品构成表.csv', index=0, encoding="utf-8")
df3.to_csv('./进出口商品总值表(人民币值)年度表.csv', index=0, encoding="utf-8")
#
# # 或导出到Excel
# df1.to_excel(r"C:\Users\QDM\Desktop\世界大学综合排名.xlsx", index=0)
