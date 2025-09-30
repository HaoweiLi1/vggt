#!/bin/bash

# 批量处理所有数据集文件夹的脚本
# 使用robust的batch_crop.py脚本处理training/dataset_aerialmegadepth下的所有文件夹

set -e  # 遇到错误时退出

# 配置参数
INPUT_BASE_DIR="training/dataset_aerialmegadepth"
OUTPUT_BASE_DIR="training/dataset_aerialmegadepth_cropped"
SCRIPT_PATH="batch_crop.py"

# 检查脚本是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "错误: 找不到批处理脚本 $SCRIPT_PATH"
    exit 1
fi

# 检查输入目录是否存在
if [ ! -d "$INPUT_BASE_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_BASE_DIR"
    exit 1
fi

# 创建输出基础目录
mkdir -p "$OUTPUT_BASE_DIR"

echo "=== 批量处理数据集文件夹 ==="
echo "输入基础目录: $INPUT_BASE_DIR"
echo "输出基础目录: $OUTPUT_BASE_DIR"
echo "处理脚本: $SCRIPT_PATH"
echo

# 获取所有需要处理的文件夹
echo "扫描需要处理的文件夹..."
folders_to_process=()

for folder in "$INPUT_BASE_DIR"/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        # 跳过非数字文件夹（如果有的话）
        if [[ "$folder_name" =~ ^[0-9]+$ ]]; then
            folders_to_process+=("$folder_name")
        fi
    fi
done

# 显示找到的文件夹
echo "找到 ${#folders_to_process[@]} 个文件夹需要处理:"
for folder in "${folders_to_process[@]}"; do
    echo "  - $folder"
done
echo

# 开始处理
total_folders=${#folders_to_process[@]}
current_folder=0

for folder in "${folders_to_process[@]}"; do
    current_folder=$((current_folder + 1))
    input_dir="$INPUT_BASE_DIR/$folder"
    output_dir="$OUTPUT_BASE_DIR/$folder"
    
    echo "[$current_folder/$total_folders] 处理文件夹: $folder"
    echo "  输入目录: $input_dir"
    echo "  输出目录: $output_dir"
    
    # 检查输入目录中是否有文件
    file_count=$(find "$input_dir" -type f \( -name "*.jpg" -o -name "*.exr" -o -name "*.npz" \) | wc -l)
    if [ "$file_count" -eq 0 ]; then
        echo "  警告: 输入目录中没有找到支持的文件，跳过"
        echo
        continue
    fi
    
    echo "  找到 $file_count 个文件需要处理"
    
    # 执行处理
    start_time=$(date +%s)
    if python3 "$SCRIPT_PATH" "$input_dir" "$output_dir"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "  ✓ 处理完成，耗时: ${duration}秒"
        
        # 验证输出文件数量
        output_file_count=$(find "$output_dir" -type f | wc -l)
        echo "  输出文件数量: $output_file_count"
        
        if [ "$output_file_count" -eq "$file_count" ]; then
            echo "  ✓ 文件数量验证通过"
        else
            echo "  ⚠️  文件数量不匹配 (输入: $file_count, 输出: $output_file_count)"
        fi
    else
        echo "  ✗ 处理失败"
        echo "  错误: 处理文件夹 $folder 时出现错误"
        # 可以选择继续处理其他文件夹或者退出
        # exit 1  # 取消注释这行会在出错时退出
    fi
    
    echo
done

echo "=== 批量处理完成 ==="
echo "处理了 $total_folders 个文件夹"
echo "输出目录: $OUTPUT_BASE_DIR"

# 显示最终统计信息
echo
echo "=== 最终统计 ==="
for folder in "${folders_to_process[@]}"; do
    input_dir="$INPUT_BASE_DIR/$folder"
    output_dir="$OUTPUT_BASE_DIR/$folder"
    
    if [ -d "$output_dir" ]; then
        input_count=$(find "$input_dir" -type f \( -name "*.jpg" -o -name "*.exr" -o -name "*.npz" \) | wc -l)
        output_count=$(find "$output_dir" -type f | wc -l)
        echo "文件夹 $folder: 输入 $input_count 文件, 输出 $output_count 文件"
    else
        echo "文件夹 $folder: 未处理或处理失败"
    fi
done

echo
echo "全部处理完成！"