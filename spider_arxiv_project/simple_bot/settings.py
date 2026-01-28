# simple_bot/settings.py

BOT_NAME = "simple_bot"
SPIDER_MODULES = ["simple_bot.spiders"]
NEWSPIDER_MODULE = "simple_bot.spiders"

# 1. 启用我们的保存管道
ITEM_PIPELINES = {
   "simple_bot.pipelines.SavePdfPipeline": 300,
}

# 2. 伪装 UA (必须)
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# 3. 礼貌爬取
ROBOTSTXT_OBEY = False
DOWNLOAD_DELAY = 1.5      # 每个请求间隔 1.5 秒
CONCURRENT_REQUESTS = 5   # 同时下载 5 个

# 4. 日志只看重要的
LOG_LEVEL = 'INFO'