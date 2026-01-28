# simple_bot/spiders/arxiv.py
import scrapy
from simple_bot.items import ArxivDocItem

class ArxivSpider(scrapy.Spider):
    name = "arxiv"
    # 搜索：半导体 + 2024年 + 物理类
    start_urls = [
        'https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=semiconductor&terms-0-field=title&classification-physics=y&classification-include_cross_list=include&date-filter_by=specific_year&date-year=2024&date-filter_by=past_12_months&abstracts=show&size=50&order=-announced_date_first'
    ]

    def parse(self, response):
        print(f"【状态】正在解析列表页: {response.url}")
        papers = response.css('li.arxiv-result')

        if not papers:
            print("【警告】未找到论文，可能被反爬或选择器失效。")

        for paper in papers:
            # 1. 提取标题
            title = paper.css('p.title::text').get().strip()
            
            # 2. 提取日期
            date_texts = paper.css('p.is-size-7::text').getall()
            date_str = date_texts[1].strip() if len(date_texts) > 1 else "Unknown"

            # 3. 提取 PDF 链接
            pdf_link = paper.css('p.list-title a[href*="pdf"]::attr(href)').get()

            if pdf_link:
                print(f"  -> 发现论文: {title[:30]}... | 准备下载")
                
                # 构造 Item
                item = ArxivDocItem()
                item['title'] = title
                item['url'] = pdf_link
                item['publish_date'] = date_str
                
                # 发起这个 PDF 的下载请求
                yield scrapy.Request(
                    url=pdf_link,
                    callback=self.handle_pdf_download,
                    meta={'item': item},
                    dont_filter=True
                )

    def handle_pdf_download(self, response):
        """
        这里处理 PDF 的二进制响应
        """
        item = response.meta['item']
        
        # 简单检查 Content-Type
        if b'%PDF' in response.body[:10] or 'pdf' in response.headers.get('Content-Type', b'').decode('utf-8').lower():
            # 把二进制内容塞进 Item，交给 Pipeline 去保存
            item['file_content'] = response.body
            
            # 简单的文件名清洗
            safe_title = "".join([c for c in item['title'] if c.isalnum() or c in " -_"])[:50]
            item['file_name'] = f"{safe_title}.pdf"
            
            yield item
        else:
            print(f"【失败】下载的不是 PDF: {item['url']}")