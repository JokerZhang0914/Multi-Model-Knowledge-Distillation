import os
from moviepy import VideoFileClip
from moviepy.video.fx import Crop 

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

# --- 配置参数 ---
if __name__ == "__main__":
    
    pre_dir = "./data/肿瘤医院_1"
    # 输入文件夹路径
    input_dir = pre_dir + "/大肠息肉内镜视频2"
    
    # 输出文件夹路径
    output_dir = pre_dir + "/视频cut"
    
    # 裁剪区域配置
    crop_settings = {
        'x1': 700,
        'y1': 0,
        'x2': 1860,
        'y2': 1080
    }

    # 执行函数
    crop_videos(input_dir, output_dir, crop_settings)