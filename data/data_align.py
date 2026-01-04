import os
import shutil
import re
import random
import pandas as pd
from math import ceil

def aligned_images_zhongliu(input_dir, output_root_dir):
    """
    整理图片：按视频来源分组 -> 【随机打乱】 -> 每50张切分(尾数处理) -> 重命名并移动
    """
    
    # 1. 检查输入目录
    if not os.path.exists(input_dir):
        print(f"error：输入目录不存在 -> {input_dir}")
        return

    # 2. 扫描并解析所有图片
    # 结构: video_dict = { 视频编号n: [(原始文件名, 原始序号m), ...], ... }
    video_dict = {}
    
    files = os.listdir(input_dir)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    print("正在扫描文件...")
    
    for fname in files:
        if not fname.lower().endswith(valid_extensions):
            continue
            
        # 解析文件名 1_n_m.jpg
        match = re.match(r'1_(\d+)_(\d+)', fname)
        if match:
            video_n = int(match.group(1)) # 视频编号
            frame_m = int(match.group(2)) # 原始序号
            
            if video_n not in video_dict:
                video_dict[video_n] = []
            
            video_dict[video_n].append((fname, frame_m))
            
    print(f"共识别到 {len(video_dict)} 个视频来源的图片。")

    # 3. 按视频编号排序处理 (虽然图片要打乱，但处理视频的顺序最好还是固定的，方便观察)
    sorted_video_ids = sorted(video_dict.keys())
    
    # 全局组计数器 (用来生成 t_patient1, t_patient2...)
    global_group_counter = 1
    
    for vid in sorted_video_ids:
        # 获取该视频下的所有图片列表
        images = video_dict[vid]
        total_imgs = len(images)
        
        print(f"\n正在处理视频 n={vid} (共 {total_imgs} 张)...")
        
        # --- 【核心修改】随机打乱 ---
        # 原代码是按序号排序: sorted(images, key=lambda x: x[1])
        # 现在改为原地随机打乱
        random.shuffle(images)
        print(f"  -> 已随机打乱图片顺序")
        # -------------------------
        
        # --- 分组算法 ---
        groups = []
        chunk_size = 50
        min_remainder = 30
        
        # 初始切分
        for i in range(0, total_imgs, chunk_size):
            groups.append(images[i:i + chunk_size])
            
        # 检查最后一组的数量
        if len(groups) > 1:
            last_group = groups[-1]
            if len(last_group) < min_remainder:
                # 如果最后一组不足30张，合并到倒数第二组
                prev_group = groups[-2]
                prev_group.extend(last_group)
                # 删除最后一组
                groups.pop()
                print(f"  -> 尾部合并: 最后 {len(last_group)} 张并入上一组，上一组现变为 {len(prev_group)} 张")

        # --- 创建文件夹并移动/重命名 ---
        for group in groups:
            # 定义当前组的文件夹名称
            folder_name = f"t_patient{global_group_counter}"
            folder_path = os.path.join(output_root_dir, folder_name)
            
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # 处理组内图片
            # index 是新生成的顺序 (patient_1, patient_2...)
            for index, (original_fname, _) in enumerate(group, start=1):
                
                # 构造新文件名: patient{组号}_{组内序号}.jpg
                # 例如: patient1_1.jpg, patient1_2.jpg...
                new_fname = f"patient{global_group_counter}_{index}.jpg"
                
                src = os.path.join(input_dir, original_fname)
                dst = os.path.join(folder_path, new_fname)
                
                try:
                    # shutil.copy2 支持中文路径，且保留文件元数据
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"复制失败: {original_fname} -> {e}")

            print(f"  -> 生成组: {folder_name} (包含 {len(group)} 张)")
            
            # 只有处理完一组，计数器才+1
            global_group_counter += 1

    print("\n" + "="*30)
    print("整理完成！")
    print(f"总共生成了 {global_group_counter - 1} 个文件夹。")
    print(f"保存路径: {os.path.abspath(output_root_dir)}")

def extract_prefix(filename):
    """
    自动提取前缀。
    逻辑：文件名去除后缀后，匹配末尾的连续数字，数字前的部分即为前缀。
    例如: "WL (20)_256.jpg" -> 匹配末尾 "256" -> 返回 "WL (20)_"
    """
    # 1. 去除后缀 (如 .jpg)
    name_no_ext = os.path.splitext(filename)[0]
    
    # 2. 正则匹配：找到结尾的数字
    # r'(.*?)(\d+)$' 
    # group(1) 是非贪婪匹配的前缀，group(2) 是结尾的数字
    match = re.search(r'^(.*?)(\d+)$', name_no_ext)
    
    if match:
        return match.group(1) # 返回前缀部分
    else:
        return None # 如果没有以数字结尾，则视为不符合规则

# --- 主处理函数 ---
def aligned_images_gongkai(input_dir, output_dir, start_group_index):
    
    if not os.path.exists(input_dir):
        print(f"❌ 错误：输入目录不存在 -> {input_dir}")
        return

    print(f"正在扫描目录: {input_dir} ...")
    
    # --- 【修改】第一步：自动归类所有图片 ---
    # 结构: files_by_prefix = { "前缀A": [文件1, 文件2...], "前缀B": [...] }
    files_by_prefix = {}
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    all_files = os.listdir(input_dir)
    count_valid = 0
    
    for fname in all_files:
        if fname.lower().endswith(valid_extensions):
            prefix = extract_prefix(fname)
            
            if prefix is not None:
                if prefix not in files_by_prefix:
                    files_by_prefix[prefix] = []
                files_by_prefix[prefix].append(fname)
                count_valid += 1
            else:
                # 可以选择打印那些不符合命名规范的文件
                # print(f"跳过无数字结尾文件: {fname}")
                pass
                
    print(f"✅ 扫描完成。共找到 {len(files_by_prefix)} 种不同的前缀，合计 {count_valid} 张图片。")
    print("-" * 30)

    # --- 【修改】第二步：按前缀组依次处理 ---
    # 按照前缀字母顺序排序处理，保证每次运行的处理顺序（除了图片打乱）是一致的
    sorted_prefixes = sorted(files_by_prefix.keys())
    
    # 全局计数器 n，从用户指定的 start_group_index 开始
    current_group_n = start_group_index 
    
    for prefix in sorted_prefixes:
        file_list = files_by_prefix[prefix]
        total_files = len(file_list)
        
        print(f"正在处理前缀组: [{prefix}] (共 {total_files} 张)")

        # 1. 随机打乱 (同之前逻辑)
        random.shuffle(file_list)

        # 2. 分组逻辑 (同之前逻辑: 80一组，余数<50合并)
        groups = []
        chunk_size = 60
        min_remainder = 35
        
        for i in range(0, total_files, chunk_size):
            groups.append(file_list[i : i + chunk_size])
            
        if len(groups) > 1:
            last_group = groups[-1]
            if len(last_group) < min_remainder:
                print(f"  ⚠️ 尾部合并: 最后 {len(last_group)} 张并入上一组")
                prev_group = groups[-2]
                prev_group.extend(last_group)
                groups.pop()

        # 3. 创建文件夹并移动 (同之前逻辑)
        for group in groups:
            folder_name = f"t_patient{current_group_n}"
            folder_path = os.path.join(output_dir, folder_name)
            
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # 组内文件重命名
            for index, filename in enumerate(group, start=1):
                ext = os.path.splitext(filename)[1]
                # 命名格式: patient{n}_{m}.jpg
                new_filename = f"patient{current_group_n}_{index}{ext}"
                
                src_path = os.path.join(input_dir, filename)
                dst_path = os.path.join(folder_path, new_filename)
                
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(f"  ❌ 复制失败: {filename} -> {e}")
            
            # 处理完一组，组号+1
            current_group_n += 1
            
        print(f"  -> 前缀 [{prefix}] 处理完毕。current_group_n:{current_group_n-1}\n")

    print("="*30)
    print("全部任务完成！")
    print(f"生成的文件夹范围: f_patient{start_group_index} 到 f_patient{current_group_n - 1}")
    print(f"输出目录: {output_dir}")

def aligned_images_zhongshan(excel_path, output_root_dir,t_count,f_count):
    print(f"正在读取文件: {excel_path}")
    
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"❌ 读取 Excel 失败: {e}")
        return

    # 1. 确保目标根目录存在
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
        print(f"创建输出根目录: {output_root_dir}")

    # 2. 确保 J 列存在 (path)
    if 'image_folder_path' not in df.columns:
        df['image_folder_path'] = None   
    
    # 原始数据根目录
    source_root = r"H:\中山医院"
    
    # 4. 遍历处理
    total_rows = len(df)
    processed_count = 0

    for index, row in df.iterrows():
        # --- 检查 file_exists (I列) ---
        # 注意: 这里的 'file_exists' 列名需要与上一步生成的 Excel 一致
        # 有些可能是 1, 1.0 或 True
        exists_val = row.get('file_exists', 0)
        try:
            is_exists = int(exists_val) == 1
        except:
            is_exists = False

        if not is_exists:
            continue  # 如果文件不存在，直接跳过
        
        processed_count += 1
        print(f"\n[{processed_count}] 正在处理行 {index + 1}...")

        # --- 构建源路径 ---
        raw_date = row.get('检查日期')
        tech_id = str(row.get('医技号', '')).strip()
        serial_id = str(row.get('医技检查流水号', '')).strip()
        label_val = row.get('label', 0) # 获取 label (0 或 1)
        
        # 日期格式化
        date_folder = ""
        try:
            if isinstance(raw_date, pd.Timestamp):
                date_folder = raw_date.strftime('%Y%m%d')
            else:
                date_str = str(raw_date).split(' ')[0]
                date_folder = date_str.replace('/', '').replace('-', '')
        except:
            pass
            
        src_dir = os.path.join(source_root, date_folder, tech_id, serial_id)
        
        if not os.path.exists(src_dir):
            print(f"  ❌ 警告: 记录显示存在，但实际路径未找到: {src_dir}")
            continue

        # --- 扫描源图片 ---
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        images = [f for f in os.listdir(src_dir) if f.lower().endswith(valid_exts)]
        
        if not images:
            print("  ⚠️ 文件夹为空，跳过")
            continue
            
        # 随机打乱
        random.shuffle(images)
        
        # --- 分组逻辑 (60一组, 尾数<40合并) ---
        groups = []
        chunk_size = 60
        min_remainder = 40
        
        for i in range(0, len(images), chunk_size):
            groups.append(images[i : i + chunk_size])
            
        if len(groups) > 1:
            if len(groups[-1]) < min_remainder:
                prev = groups[-2]
                prev.extend(groups[-1])
                groups.pop()
        
        # --- 确定当前病人的前缀类型和起始编号 ---
        # 如果 label 是 1 -> t_patient
        # 如果 label 是 0 -> f_patient
        try:
            is_positive = int(float(label_val)) == 1
        except:
            is_positive = False # 默认 0
            
        current_paths = [] # 记录当前病人生成的文件夹路径

        # --- 移动并重命名 ---
        for group in groups:
            # 确定文件夹名
            if is_positive:
                folder_name = f"t_patient{t_count}"
                # 更新组号用于下一组
                current_group_num = t_count
                t_count += 1
            else:
                folder_name = f"f_patient{f_count}"
                current_group_num = f_count
                f_count += 1
            
            target_dir = os.path.join(output_root_dir, folder_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            # 记录该文件夹路径
            current_paths.append(target_dir)
            
            # 复制并重命名图片
            for img_idx, img_name in enumerate(group, start=1):
                # 命名格式: patient{n}_{m}.jpg (注意这里不带 t_ 或 f_)
                # 保持原后缀
                ext = os.path.splitext(img_name)[1]
                new_name = f"patient{current_group_num}_{img_idx}{ext}"
                
                src_file = os.path.join(src_dir, img_name)
                dst_file = os.path.join(target_dir, new_name)
                
                try:
                    shutil.copy2(src_file, dst_file)
                except Exception as e:
                    print(f"  复制失败: {img_name} -> {e}")
            
            print(f"  -> 生成: {folder_name} ({len(group)} 张)")

        # --- 写入 Excel J列 ---
        # 将生成的文件夹路径用分号连接 (如果一个病人生成了多个文件夹)
        paths_str = "; ".join(current_paths)
        df.at[index, 'image_folder_path'] = paths_str

    # 5. 保存 Excel
    try:
        output_excel_path = excel_path.replace(".xlsx", "_processed.xlsx")
        df.to_excel(output_excel_path, index=False)
        print("\n" + "="*30)
        print("✅ 全部完成！Excel 已更新。")
        print(f"输出目录: {output_root_dir}")
        print(f"Excel路径: {output_excel_path}")
        print(f"生成的 t_patient (阳性) 总数: {t_count - 1}")
        print(f"生成的 f_patient (阴性) 总数: {f_count - 1}")
    except Exception as e:
        print(f"❌ 保存 Excel 失败: {e}")

# --- 配置区域 ---
if __name__ == "__main__":
    
    """
    pre_dir = "./data/肿瘤医院_1"
    
    # 原始图片所在位置
    input_frame_dir = os.path.join(pre_dir, "frame")
    
    # 整理后的目标根目录
    output_dataset_dir = os.path.join(pre_dir, "dataset_aligned")
    
    # 为了保证随机性，每次运行结果都不一样
    aligned_images_zhongliu(input_frame_dir, output_dataset_dir)
    """

    """
    # 公开数据集\
    # 输入目录 (注意：Windows路径建议在引号前加 r，或者把 \ 改为 /)
    input_directory = r"D:\公开数据集\2_adenomatous"
    
    # 输出目录 (你想存在哪里？这里假设存在同级的 output 文件夹中，你可以修改)
    output_directory = r"D:\公开数据集\2_adenomatous_aligned"
    
    # 起始组编号 n 
    start_n = 835

    
    aligned_images_gongkai(
        input_directory, 
        output_directory, 
        start_n
    )
    """

    input_excel_path = r"E:\AAA_joker\本科毕设\code\data\数据整理_内镜肠镜病理（2020年）_厦门.xlsx"
    
    # 输出图片根目录
    output_aligned_dir = r"H:\中山医院_aligned"
    
    t_count = 1555
    f_count = 621

    # 如果原始数据处理的 Excel 名字不同，请修改 input_excel_path
    if os.path.exists(input_excel_path):
        aligned_images_zhongshan(input_excel_path, output_aligned_dir,t_count,f_count)
    else:
        print("找不到输入文件，请检查文件名是否正确")