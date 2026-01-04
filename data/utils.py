import os
from moviepy import VideoFileClip
from moviepy.video.fx import Crop 
import cv2
import numpy as np
import sys

def crop_videos(input_folder, output_folder, crop_coords):
    """
    读取指定文件夹视频，裁剪并按顺序重命名保存。
    """
    
    # 1. 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    # 2. 获取输入文件夹内的所有文件，并过滤出视频文件
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
    # 按照文件名排序
    files.sort()

    print(f"共发现 {len(files)} 个视频文件，开始处理...")

    # 3. 循环处理每个视频
    for index, filename in enumerate(files, start=54):
        input_path = os.path.join(input_folder, filename)
        
        # 设定输出文件名
        output_filename = f"{index}.mp4"
        output_path = os.path.join(output_folder, output_filename)
        
        try:
            print(f"[{index}/{len(files)}] 正在处理: {filename} -> {output_filename}")
            
            # 加载视频
            clip = VideoFileClip(input_path)
            
            # 使用 with_effects 和 Crop 类进行裁剪
            cropped_clip = clip.with_effects([
                Crop(
                    x1=crop_coords['x1'], 
                    y1=crop_coords['y1'], 
                    x2=crop_coords['x2'], 
                    y2=crop_coords['y2']
                )
            ])
            
            # 写入文件
            # 【修复点】：删除了 verbose 参数
            # logger=None 已经足以屏蔽进度条输出
            cropped_clip.write_videofile(
                output_path, 
                codec='libx264', 
                audio_codec='aac',
                temp_audiofile=f'temp-audio-{index}.m4a',
                remove_temp=True,
                logger=None 
            )
            
            # 释放内存
            clip.close()
            cropped_clip.close()
            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            # 如果出错，尝试显式关闭以释放资源
            try: clip.close() 
            except: pass

    print("所有视频处理完成！")


def cv_imwrite(filename, img):
    """
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
    
    # 视频分帧
    input_video_dir = pre_dir + "/视频cut"
    output_frame_dir = pre_dir + "/frame"

    # 裁剪视频
    # input_dir = pre_dir + "/大肠息肉内镜视频2"
    # output_dir = pre_dir + "/视频cut"

    # 裁剪区域配置
    """
    crop_settings = {
        'x1': 700,
        'y1': 0,
        'x2': 1860,
        'y2': 1080
    }
    """

    # 执行函数
    # crop_videos(input_dir, output_dir, crop_settings)

    extract_frames_final(input_video_dir, output_frame_dir)