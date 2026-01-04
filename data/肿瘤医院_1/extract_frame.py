import cv2
import os
import numpy as np
import sys

def cv_imwrite(filename, img):
    """
    【核心修复】
    自定义的图片写入函数，支持中文路径。
    原理：先用 opencv 编码，再用 python 文件流写入。
    """
    try:
        # 指定 jpg 质量为 95
        cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(filename)
        return True
    except Exception as e:
        print(f"写入异常: {e}")
        return False

def extract_frames_final(input_dir, output_dir, frame_interval=3):
    # 1. 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"输入路径: {os.path.abspath(input_dir)}")
    print(f"输出路径: {os.path.abspath(output_dir)}")

    # 2. 获取视频文件
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
    # 过滤掉系统临时文件，只留视频
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    # 排序
    try:
        video_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        video_files.sort()

    print(f"共发现 {len(video_files)} 个视频文件，开始处理...")
    
    total_saved = 0

    # 3. 循环处理
    for video_filename in video_files:
        video_path = os.path.join(input_dir, video_filename)
        
        # 获取 n (视频编号)
        fname_no_ext = os.path.splitext(video_filename)[0]
        if not fname_no_ext.isdigit():
             print(f"跳过非数字命名文件: {video_filename}")
             continue
        
        video_n = int(fname_no_ext)
        print(f"正在处理: {video_filename} (n={video_n})", end='... ')

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("无法打开")
            continue

        frame_count_in_video = 0
        saved_count_m = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每隔 frame_interval 帧保存一次
            if frame_count_in_video % frame_interval == 0:
                saved_count_m += 1
                
                # 构造文件名 1_n_m.jpg
                output_filename = f"1_{video_n}_{saved_count_m}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                # --- 使用支持中文路径的写入函数 ---
                success = cv_imwrite(output_path, frame)

            frame_count_in_video += 1

        cap.release()
        print(f"生成 {saved_count_m} 张图。")
        total_saved += saved_count_m

    print("-" * 30)
    print(f"共保存 {total_saved} 张图片。")
    print(f"请检查文件夹: {output_dir}")

if __name__ == "__main__":
    pre_dir = "./data/肿瘤医院_1"
    
    # 确保这里的路径完全正确
    input_video_dir = pre_dir + "/视频cut"
    output_frame_dir = pre_dir + "/frame"
    
    extract_frames_final(input_video_dir, output_frame_dir)