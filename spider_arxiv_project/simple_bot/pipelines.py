# simple_bot/pipelines.py
import os

class SavePdfPipeline:
    def open_spider(self, spider):
        # 爬虫启动时创建文件夹
        self.save_dir = "downloaded_pdfs"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"【系统】创建下载目录: {os.path.abspath(self.save_dir)}")

    def process_item(self, item, spider):
        if 'file_content' in item and item['file_content']:
            file_path = os.path.join(self.save_dir, item['file_name'])
            
            with open(file_path, 'wb') as f:
                f.write(item['file_content'])
            
            print(f"✅ [保存成功] {item['file_name']}")
            
            # 释放内存，防止后续步骤占用
            del item['file_content']
            
        return item