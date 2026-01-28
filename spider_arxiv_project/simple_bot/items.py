# simple_bot/items.py
import scrapy

class ArxivDocItem(scrapy.Item):
    title = scrapy.Field()
    url = scrapy.Field()
    publish_date = scrapy.Field()
    # 用于在 Pipeline 里保存文件的二进制数据
    file_content = scrapy.Field()
    # 文件名
    file_name = scrapy.Field()