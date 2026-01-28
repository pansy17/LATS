import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3

# 关闭 verify=False 时的警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= 配置区域 =================
# 填入你的 Token
API_TOKEN = ""

# 这是你日志中显示已经完成解析的 Batch ID，直接填入即可
BATCH_ID = "83d5326d-ca88-4e83-bfeb-e1b840fef41f"

# 结果保存路径
OUTPUT_DIR = r"D:\a_job\y1\project+train\llm_learn\my_rag\pdf2md\downloaded_pdfs\trans_md"
# ===========================================

headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

def get_robust_session():
    """创建一个带有重试机制和长连接的 Session"""
    session = requests.Session()
    retries = Retry(
        total=5,                # 最大重试次数
        backoff_factor=1,       # 失败后等待时间 1s, 2s, 4s...
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=frozenset(['GET', 'POST'])
    )
    # 挂载适配器到 http 和 https
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def rescue_downloads():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    url = f"https://mineru.net/api/v4/extract-results/batch/{BATCH_ID}"
    print(f"正在查询 Batch ID: {BATCH_ID} 的结果...")

    session = get_robust_session()

    try:
        # 获取任务列表
        res = session.get(url, headers=headers, verify=False) # 获取列表也不验证SSL
        data = res.json()
        
        if data.get("code") != 0:
            print(f"查询失败: {data.get('msg')}")
            return

        extract_results = data["data"]["extract_result"]
        print(f"共找到 {len(extract_results)} 个文件记录。")

        for item in extract_results:
            file_name = item["file_name"]
            state = item["state"]
            
            # 只处理完成的任务
            if state == "done":
                zip_url = item.get("full_zip_url")
                if not zip_url:
                    print(f"[跳过] {file_name} 虽然完成但没有下载链接")
                    continue
                
                save_name = f"{file_name}_result.zip"
                save_path = os.path.join(OUTPUT_DIR, save_name)

                # 检查文件是否已经下载成功过（避免重复下载）
                if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
                    print(f"[已存在] {file_name} 跳过")
                    continue

                print(f"正在下载: {file_name} ...")
                
                try:
                    # 关键修改：verify=False 跳过 SSL 验证，stream=True 流式下载
                    with session.get(zip_url, stream=True, verify=False, timeout=60) as r:
                        r.raise_for_status()
                        with open(save_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    print(f"   -> [下载成功]")
                except Exception as e:
                    print(f"   -> [下载再次失败] {e}")
            else:
                print(f"[未完成] {file_name} 状态: {state}")

    except Exception as e:
        print(f"脚本执行异常: {e}")

if __name__ == "__main__":
    rescue_downloads()