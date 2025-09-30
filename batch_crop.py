#!/usr/bin/env python3
"""
批量图像裁剪脚本
支持JPG、EXR、NPZ三种文件格式的center crop处理
将1920x1080的文件从中心裁剪到518x518
"""

import os
import sys
import numpy as np
from PIL import Image
import OpenEXR
import Imath
from pathlib import Path
from tqdm import tqdm

def center_crop_jpg(input_path, output_path, target_size=518):
    """裁剪JPG图像"""
    try:
        img = Image.open(input_path)
        width, height = img.size
        
        # 计算中心裁剪区域
        center_x = width // 2
        center_y = height // 2
        half_size = target_size // 2
        
        left = center_x - half_size
        top = center_y - half_size
        right = center_x + half_size
        bottom = center_y + half_size
        
        # 裁剪并保存
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"处理JPG文件失败 {input_path}: {e}")
        return False

def center_crop_exr(input_path, output_path, target_size=518):
    """裁剪EXR文件"""
    try:
        # 读取EXR文件
        exr_file = OpenEXR.InputFile(input_path)
        header = exr_file.header()
        
        # 获取尺寸和通道
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        channels = list(header['channels'].keys())
        
        # 读取所有通道数据
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channel_data = {}
        
        for channel in channels:
            channel_str = exr_file.channel(channel, FLOAT)
            channel_array = np.frombuffer(channel_str, dtype=np.float32)
            channel_array = channel_array.reshape((height, width))
            channel_data[channel] = channel_array
        
        exr_file.close()
        
        # 计算中心裁剪区域
        center_x = width // 2
        center_y = height // 2
        half_size = target_size // 2
        
        start_x = center_x - half_size
        end_x = center_x + half_size
        start_y = center_y - half_size
        end_y = center_y + half_size
        
        # 裁剪所有通道
        cropped_data = {}
        for channel, data in channel_data.items():
            cropped_data[channel] = data[start_y:end_y, start_x:end_x]
        
        # 创建输出EXR文件
        output_header = OpenEXR.Header(target_size, target_size)
        channel_dict = {}
        for channel in channels:
            channel_dict[channel] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        output_header['channels'] = channel_dict
        
        # 写入EXR文件
        exr_output = OpenEXR.OutputFile(output_path, output_header)
        write_data = {}
        for channel in channels:
            write_data[channel] = cropped_data[channel].astype(np.float32).tobytes()
        
        exr_output.writePixels(write_data)
        exr_output.close()
        return True
        
    except Exception as e:
        print(f"处理EXR文件失败 {input_path}: {e}")
        return False

def center_crop_npz(input_path, output_path, original_size, target_size=518):
    """调整NPZ文件中的相机内参"""
    try:
        # 加载NPZ文件
        data = np.load(input_path)
        output_data = {}
        
        for key in data.keys():
            output_data[key] = data[key].copy()
        
        # 调整intrinsics矩阵
        if 'intrinsics' in data:
            intrinsics = data['intrinsics'].copy()
            
            # 计算center crop的偏移
            orig_width, orig_height = original_size
            center_x = orig_width // 2
            center_y = orig_height // 2
            half_size = target_size // 2
            
            crop_start_x = center_x - half_size
            crop_start_y = center_y - half_size
            
            # 调整主点坐标
            intrinsics[0, 2] = intrinsics[0, 2] - crop_start_x  # cx
            intrinsics[1, 2] = intrinsics[1, 2] - crop_start_y  # cy
            
            output_data['intrinsics'] = intrinsics
        
        # 保存调整后的数据
        np.savez_compressed(output_path, **output_data)
        return True
        
    except Exception as e:
        print(f"处理NPZ文件失败 {input_path}: {e}")
        return False

def get_image_size(file_path):
    """获取图像文件的尺寸"""
    try:
        if file_path.suffix.lower() in ['.jpg', '.jpeg']:
            img = Image.open(file_path)
            return img.size  # (width, height)
        elif file_path.suffix.lower() == '.exr':
            exr_file = OpenEXR.InputFile(str(file_path))
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            exr_file.close()
            return (width, height)
    except Exception as e:
        print(f"获取图像尺寸失败 {file_path}: {e}")
        return None

def process_file(input_file, output_file, original_size=None):
    """根据文件扩展名选择处理方法"""
    ext = input_file.suffix.lower()
    
    if ext == '.jpg' or ext == '.jpeg':
        return center_crop_jpg(str(input_file), str(output_file))
    elif ext == '.exr':
        return center_crop_exr(str(input_file), str(output_file))
    elif ext == '.npz':
        if original_size is None:
            print(f"NPZ文件需要原始图像尺寸信息: {input_file}")
            return False
        return center_crop_npz(str(input_file), str(output_file), original_size)
    else:
        print(f"不支持的文件格式: {ext}")
        return False

def batch_crop(input_dir, output_dir):
    """批量处理文件夹中的所有支持文件"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的文件扩展名
    supported_extensions = {'.jpg', '.jpeg', '.exr', '.npz'}
    
    # 收集所有文件并按基础名称分组
    all_files = {}
    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            # 获取基础文件名（去掉扩展名）
            base_name = file_path.stem
            if base_name not in all_files:
                all_files[base_name] = {}
            all_files[base_name][file_path.suffix.lower()] = file_path
    
    if not all_files:
        print(f"在目录 {input_dir} 中没有找到支持的文件")
        return
    
    print(f"找到 {len(all_files)} 个文件组需要处理")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 统计处理结果
    success_count = 0
    failed_files = []
    total_files = 0
    
    # 按组处理文件
    for base_name, file_group in tqdm(all_files.items(), desc="处理文件组"):
        # 首先尝试获取图像尺寸
        original_size = None
        
        # 优先从JPG文件获取尺寸
        for ext in ['.jpg', '.jpeg', '.exr']:
            if ext in file_group:
                original_size = get_image_size(file_group[ext])
                if original_size:
                    break
        
        # 处理该组的所有文件
        for ext, input_file in file_group.items():
            total_files += 1
            output_file = output_path / input_file.name
            
            success = process_file(input_file, output_file, original_size)
            if success:
                success_count += 1
            else:
                failed_files.append(str(input_file))
    
    # 输出结果统计
    print(f"\n处理完成！")
    print(f"成功处理: {success_count}/{total_files} 个文件")
    
    if failed_files:
        print(f"失败的文件:")
        for failed_file in failed_files:
            print(f"  - {failed_file}")

def main():
    if len(sys.argv) != 3:
        print("用法: python batch_crop.py <输入目录> <输出目录>")
        print("示例: python batch_crop.py training/dataset_aerialmegadepth/0000 cropped_output")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    batch_crop(input_dir, output_dir)

if __name__ == "__main__":
    main()