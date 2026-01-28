# run.py
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
import os

# 确保能找到 settings
os.environ.setdefault('SCRAPY_SETTINGS_MODULE', 'simple_bot.settings')

if __name__ == "__main__":
    print("=== 🚀 启动简易版半导体爬虫 ===")
    
    # 加载 settings.py 的配置
    settings = get_project_settings()
    process = CrawlerProcess(settings)
    
    # 指定要运行的爬虫名字 (和 spiders/arxiv.py 里的 name 一致)
    process.crawl("arxiv")
    
    process.start()
    print("=== ✅ 爬取任务结束 ===")