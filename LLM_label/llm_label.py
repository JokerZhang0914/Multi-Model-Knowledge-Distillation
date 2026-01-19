import time
import os
import pandas as pd
from openai import OpenAI
from openai import RateLimitError, APIError, APIConnectionError

# ===========================================================
# 配置 API
# ===========================================================

client = OpenAI(
    api_key="sk-ctKXfaiULtRiaWSyWbZ0Pu1tzd3c5IMbY6Y0dBsDuXBWAbtj", 
    base_url="https://api.moonshot.cn/v1",
)

SYSTEM_PROMPT = "你是一名专业的消化内镜医师。"

USER_TEMPLATE = (
    "根据下面提供的文本，判断内容是否提及‘息肉’。"
    "平坦或隆起样病灶也属于息肉。"
    "如果有息肉，回答 1；如果没有息肉，回答 0。"
    "不需要解释，只输出 1 或 0。\n\n"
    "文本如下：{}\n"
)

USER_TEMPLATE_ZHONGSHAN = (
    "根据下面提供的病理报告（包含巨检和病理诊断，同等重要），判断内容是否提及‘息肉’。\n"
    "平坦或隆起样病灶也属于息肉。\n"
    "如果有息肉，回答 1；如果没有息肉，回答 0。\n"
    "不需要解释，只输出 1 或 0。\n\n"
    "【巨检内容】：{}\n"
    "【病理诊断】：{}\n"
)


# ===========================================================
# 调用 API 的安全函数：自动处理限流、超时、重试
# ===========================================================
def call_llm_safe(messages, max_retries=5):
    retry = 0

    while True:
        try:
            response = client.chat.completions.create(
                model="kimi-k2-turbo-preview",
                messages=messages,
                temperature=0
            )
            return response

        except RateLimitError as e:
            retry += 1
            if retry > max_retries:
                raise e
            print(f"[RateLimit] API 限流，等待 3 秒后重试... (第 {retry} 次)")
            time.sleep(3)

        except (APIError, APIConnectionError) as e:
            retry += 1
            if retry > max_retries:
                raise e
            print(f"[APIError] 网络或服务错误，等待 2 秒后重试... (第 {retry} 次)")
            time.sleep(2)

        except Exception as e:
            raise e


# ===========================================================
# 封装主评估函数
# ===========================================================
def evaluate_polyps_test(csv_path):
    df = pd.read_csv(csv_path)
    
    predictions = []
    correct = 0

    for idx, row in df.iterrows():
        text = row["report_text"]
        true_label = int(row["label"])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text)},
        ]

        # 调用 API
        response = call_llm_safe(messages)

        raw_output = response.choices[0].message.content.strip()

        # 解析为 0/1
        if raw_output.startswith("1"):
            pred = 1
        elif raw_output.startswith("0"):
            pred = 0
        else:
            pred = 1 if "1" in raw_output else 0

        predictions.append(pred)
        correct += int(pred == true_label)

        print(f"[{idx}] 真={true_label} 预测={pred} 输出='{raw_output}'")

        # 由于限制为 20 RPM，必须等待至少 3 秒
        time.sleep(3)

    total = len(df)
    accuracy = correct / total

    print("====================================")
    print(f"总样本数：{total}")
    print(f"正确数量：{correct}")
    print(f"准确率：{accuracy:.4f}")
    print("====================================")

    return accuracy

# ===========================================================
# 3. 核心处理函数 extract_label_zhongshan
# ===========================================================
def extract_label_zhongshan(excel_path):
    print(f"正在读取文件: {excel_path}")
    
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return

    # 根目录
    root_folder = r"H:\中山医院"

    # 初始化列（如果不存在）
    if 'label' not in df.columns:
        df['label'] = None
    if 'file_exists' not in df.columns:
        df['file_exists'] = 0

    total_rows = len(df)
    llm_call_count = 0 # 统计实际调用次数

    for index, row in df.iterrows():
        print(f"\n--- 处理第 {index + 1} / {total_rows} 行 ---")

        # ==========================================
        # 步骤 1: 先判断路径是否存在 (填入 I 列)
        # ==========================================
        raw_date = row.get('检查日期')
        tech_id = str(row.get('医技号', '')).strip()
        check_serial_id = str(row.get('医技检查流水号', '')).strip()
        
        # 转换日期格式 -> YYYYMMDD
        date_folder_name = ""
        try:
            if isinstance(raw_date, pd.Timestamp):
                date_folder_name = raw_date.strftime('%Y%m%d')
            else:
                # 假设字符串是 '2020/08/05' 格式
                date_str = str(raw_date).split(' ')[0]
                date_folder_name = date_str.replace('/', '').replace('-', '')
        except:
            pass

        # 构建路径
        target_path = os.path.join(root_folder, date_folder_name, tech_id, check_serial_id)
        
        # 检查存在性
        exists = 0
        if date_folder_name and tech_id and check_serial_id and os.path.exists(target_path):
            exists = 1
            print(f"  [Path] ✅ 路径存在: {target_path}")
        else:
            exists = 0
            if not (date_folder_name and tech_id and check_serial_id):
                 print(f"  [Path] ❌ 信息缺失，跳过路径检查")
            else:
                 print(f"  [Path] ❌ 路径不存在: {target_path}")

        # 更新 file_exists 列
        df.at[index, 'file_exists'] = exists

        # ==========================================
        # 步骤 2: 条件判断是否调用 API (填入 H 列)
        # ==========================================
        current_label = df.at[index, 'label']
        
        # 判断 Label 是否为空 (NaN, None, 或空字符串)
        is_label_empty = pd.isna(current_label) or str(current_label).strip() == ""

        # 【核心逻辑】：文件存在 AND 标签为空 -> 才调用 API
        if exists == 1 and is_label_empty:
            print("  [LLM] 满足条件 (文件存在且无标签)，正在调用 API...")
            
            macroscopic = str(row.get('巨检', ''))
            diagnosis = str(row.get('病理诊断', ''))
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(macroscopic, diagnosis)},
            ]

            response = call_llm_safe(messages)
            
            if response:
                raw_output = response.choices[0].message.content.strip()
                if "1" in raw_output:
                    pred = 1
                elif "0" in raw_output:
                    pred = 0
                else:
                    pred = 0
                
                print(f"  [LLM] 预测结果: {pred}")
                df.at[index, 'label'] = pred
                llm_call_count += 1
                
                # 只有调用了 API 才需要 sleep
                time.sleep(3.1)
            else:
                print("  [LLM] 调用失败")
        
        else:
            # 打印跳过原因
            if exists == 0:
                print("  [Skip] 跳过 API: 文件不存在")
            elif not is_label_empty:
                print(f"  [Skip] 跳过 API: 标签已有值 ({current_label})")

    # ==========================================
    # 保存结果
    # ==========================================
    output_path = excel_path.replace(".xlsx", "_processed.xlsx")
    df.to_excel(output_path, index=False)
    print("===========================================================")
    print(f"处理完成！实际调用 LLM 次数: {llm_call_count}")
    print(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    # csv_file = "./LLM_label/report.csv" 
    # acc = evaluate_polyps_test(csv_file)
    # print(f"返回的准确率：{acc:.4f}")
    
    # 输入文件路径 
    input_excel = r"E:\AAA_joker\本科毕设\code\data\数据整理_内镜肠镜病理（2020年）_厦门.xlsx"
    
    extract_label_zhongshan(input_excel)