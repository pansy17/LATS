import os
import time
import requests
import json
import math
from pathlib import Path

# ================= 配置区域 =================
# 请在此处填入你的 API Token
API_TOKEN = ""

# 待解析的本地 PDF 文件夹路径
INPUT_DIR = r"D:/a_job/y1/project+train/llm_learn/my_rag/pdf2md/downloaded_pdfs"

# 结果保存路径 (默认在原文件夹下的 output 目录)
OUTPUT_DIR = os.path.join(INPUT_DIR, "trans_md")

# Mineru API URL
BASE_URL = "https://mineru.net/api/v4"
BATCH_SIZE = 200  # API 限制单次最大 200 个文件

# 模型版本: 'pipeline' 或 'vlm' (根据文档推荐，vlm 效果较好)
MODEL_VERSION = "vlm"
# ===========================================

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_TOKEN}"
}

def get_files(directory):
    """扫描目录下所有 PDF 文件"""
    path = Path(directory)
    files = list(path.glob("*.pdf"))
    return [str(f) for f in files]

def apply_upload_urls(file_paths):
    """第一步：申请上传链接"""
    url = f"{BASE_URL}/file-urls/batch"
    
    files_payload = []
    for fp in file_paths:
        file_name = os.path.basename(fp)
        # 使用文件名作为 data_id (去除特殊字符以防万一，这里简单处理)
        files_payload.append({
            "name": file_name,
            "data_id": file_name
        })

    data = {
        "files": files_payload,
        "model_version": MODEL_VERSION
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        resp_json = response.json()
        if resp_json.get("code") == 0:
            return resp_json["data"]["batch_id"], resp_json["data"]["file_urls"]
        else:
            print(f"[Error] 申请上传链接失败: {resp_json.get('msg')}")
            return None, None
    except Exception as e:
        print(f"[Exception] 请求异常: {e}")
        return None, None

def upload_files(file_paths, upload_urls):
    """第二步：上传文件二进制流"""
    print(f"   > 开始上传 {len(file_paths)} 个文件...")
    for i, file_path in enumerate(file_paths):
        upload_url = upload_urls[i]
        try:
            with open(file_path, 'rb') as f:
                # 注意：上传文件不需要 Bearer Token，直接 PUT 到阿里云/S3 链接
                # 且无需设置 Content-Type，requests 会自动处理
                res = requests.put(upload_url, data=f)
                if res.status_code == 200:
                    print(f"     [成功] 上传: {os.path.basename(file_path)}")
                else:
                    print(f"     [失败] 上传: {os.path.basename(file_path)} (Code: {res.status_code})")
        except Exception as e:
            print(f"     [异常] 上传文件出错 {file_path}: {e}")

def wait_for_completion(batch_id):
    """第三步：轮询任务进度并返回结果链接"""
    url = f"{BASE_URL}/extract-results/batch/{batch_id}"
    print(f"   > 开始轮询任务状态 (Batch ID: {batch_id})...")
    
    while True:
        try:
            response = requests.get(url, headers=headers)
            res_json = response.json()
            
            if res_json.get("code") != 0:
                print(f"   [Error] 查询进度失败: {res_json.get('msg')}")
                break

            extract_results = res_json["data"]["extract_result"]
            
            # 统计状态
            stats = {"done": 0, "running": 0, "pending": 0, "failed": 0, "waiting-file": 0}
            all_finished = True
            
            for item in extract_results:
                state = item.get("state")
                if state in stats:
                    stats[state] += 1
                # 只要有一个不是 done 或 failed，就说明任务还没全部结束
                if state not in ["done", "failed"]:
                    all_finished = False
            
            print(f"     当前进度: 完成[{stats['done']}] 进行中[{stats['running']}] 排队[{stats['pending']}] 失败[{stats['failed']}]")
            
            if all_finished:
                print("   > 本批次任务全部结束。")
                return extract_results
            
            time.sleep(5) # 每5秒查询一次
            
        except Exception as e:
            print(f"   [Exception] 轮询异常: {e}")
            time.sleep(5)

def download_results(extract_results):
    """第四步：下载解析成功的 ZIP 包"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"   > 开始下载结果到: {OUTPUT_DIR}")
    
    for item in extract_results:
        if item["state"] == "done" and item.get("full_zip_url"):
            file_name = item["file_name"]
            zip_url = item["full_zip_url"]
            save_name = f"{file_name}_result.zip"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            
            try:
                # 流式下载
                r = requests.get(zip_url, stream=True)
                if r.status_code == 200:
                    with open(save_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"     [下载完成] {save_name}")
                else:
                    print(f"     [下载失败] {file_name} HTTP {r.status_code}")
            except Exception as e:
                print(f"     [下载异常] {file_name}: {e}")
        elif item["state"] == "failed":
            print(f"     [解析失败] {item['file_name']} 原因: {item.get('err_msg')}")

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 文件夹不存在 {INPUT_DIR}")
        return

    all_files = get_files(INPUT_DIR)
    total_files = len(all_files)
    print(f"=== 扫描到 {total_files} 个 PDF 文件 ===")
    
    if total_files == 0:
        return

    # 计算需要的批次数
    num_batches = math.ceil(total_files / BATCH_SIZE)
    
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, total_files)
        batch_files = all_files[start_idx:end_idx]
        
        print(f"\n>>> 处理第 {i+1}/{num_batches} 批次 (文件 {start_idx+1} - {end_idx})")
        
        # 1. 申请链接
        batch_id, upload_urls = apply_upload_urls(batch_files)
        if not batch_id:
            print("本批次申请失败，跳过。")
            continue
            
        # 2. 上传文件
        upload_files(batch_files, upload_urls)
        
        # 3. 轮询直到结束
        results = wait_for_completion(batch_id)
        
        # 4. 下载结果
        if results:
            download_results(results)

    print("\n=== 所有任务处理完毕 ===")

if __name__ == "__main__":
    main()