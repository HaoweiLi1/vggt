import argparse
import warnings
from pathlib import Path

import Imath
import numpy as np
import OpenEXR
from PIL import Image
from tqdm import tqdm

def center_crop_jpg(input_path, output_path, target_size=518):
    img = Image.open(input_path)
    width, height = img.size
    
    center_x = width // 2
    center_y = height // 2
    half_size = target_size // 2
    
    left = center_x - half_size
    top = center_y - half_size
    right = center_x + half_size
    bottom = center_y + half_size
    
    cropped_img = img.crop((left, top, right, bottom))
    cropped_img.save(output_path, quality=95)

def center_crop_png(input_path, output_path, target_size=518):
    img = Image.open(input_path)
    width, height = img.size
    
    center_x = width // 2
    center_y = height // 2
    half_size = target_size // 2
    
    left = center_x - half_size
    top = center_y - half_size
    right = center_x + half_size
    bottom = center_y + half_size
    
    cropped_img = img.crop((left, top, right, bottom))
    cropped_img.save(output_path)

def center_crop_exr(input_path, output_path, target_size=518):
    exr_file = OpenEXR.InputFile(input_path)
    header = exr_file.header()
    
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    channels = list(header['channels'].keys())
    
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_data = {}
    
    for channel in channels:
        channel_str = exr_file.channel(channel, FLOAT)
        channel_array = np.frombuffer(channel_str, dtype=np.float32)
        channel_array = channel_array.reshape((height, width))
        channel_data[channel] = channel_array
    
    exr_file.close()
    
    center_x = width // 2
    center_y = height // 2
    half_size = target_size // 2
    
    start_x = center_x - half_size
    end_x = center_x + half_size
    start_y = center_y - half_size
    end_y = center_y + half_size
    
    cropped_data = {}
    for channel, data in channel_data.items():
        cropped_data[channel] = data[start_y:end_y, start_x:end_x]
    
    output_header = OpenEXR.Header(target_size, target_size)
    channel_dict = {}
    for channel in channels:
        channel_dict[channel] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    output_header['channels'] = channel_dict
    
    exr_output = OpenEXR.OutputFile(output_path, output_header)
    write_data = {}
    for channel in channels:
        write_data[channel] = cropped_data[channel].astype(np.float32).tobytes()
    
    exr_output.writePixels(write_data)
    exr_output.close()

def center_crop_npz(input_path, output_path, original_size, target_size=518):
    data = np.load(input_path, allow_pickle=True)
    output_data = {key: data[key].copy() for key in data.files}

    if original_size is None:
        inferred_size = None
        for candidate in ("image_size", "orig_size", "original_size"):
            if candidate in data:
                value = np.asarray(data[candidate]).reshape(-1)
                if value.size >= 2:
                    inferred_size = int(value[0]), int(value[1])
                    break
        if inferred_size is not None:
            original_size = inferred_size
        else:
            warnings.warn(
                f"Original resolution unknown for {input_path}; intrinsic center will not be adjusted.",
                RuntimeWarning,
            )

    if 'intrinsics' in data and original_size is not None:
        intrinsics = data['intrinsics'].copy()

        orig_width, orig_height = original_size
        center_x = orig_width // 2
        center_y = orig_height // 2
        half_size = target_size // 2

        crop_start_x = center_x - half_size
        crop_start_y = center_y - half_size

        intrinsics[0, 2] = intrinsics[0, 2] - crop_start_x
        intrinsics[1, 2] = intrinsics[1, 2] - crop_start_y

        output_data['intrinsics'] = intrinsics
    elif 'intrinsics' in data:
        warnings.warn(
            f"Skipping intrinsic adjustment for {input_path}; please provide RGB/EXR images to infer size.",
            RuntimeWarning,
        )

    np.savez_compressed(output_path, **output_data)

def get_image_size(file_path):
    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        img = Image.open(file_path)
        return img.size
    elif file_path.suffix.lower() == '.exr':
        exr_file = OpenEXR.InputFile(str(file_path))
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        exr_file.close()
        return (width, height)

def process_file(input_file, output_file, original_size=None):
    ext = input_file.suffix.lower()
    
    if ext in ['.jpg', '.jpeg']:
        center_crop_jpg(str(input_file), str(output_file))
    elif ext == '.png':
        center_crop_png(str(input_file), str(output_file))
    elif ext == '.exr':
        center_crop_exr(str(input_file), str(output_file))
    elif ext == '.npz':
        center_crop_npz(str(input_file), str(output_file), original_size)

def process_single_scene(input_dir, output_dir, file_types=None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    supported_extensions = {'.jpg', '.jpeg', '.exr', '.npz', '.png'}
    
    # 如果指定了文件类型，只处理这些类型
    if file_types:
        supported_extensions = {ext if ext.startswith('.') else f'.{ext}' for ext in file_types}
    
    all_files = {}
    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            base_name = file_path.stem
            if base_name not in all_files:
                all_files[base_name] = {}
            all_files[base_name][file_path.suffix.lower()] = file_path
    
    for base_name, file_group in tqdm(all_files.items()):
        original_size = None
        
        for ext in ['.jpg', '.jpeg', '.png', '.exr']:
            if ext in file_group:
                original_size = get_image_size(file_group[ext])
                if original_size:
                    break
        
        for ext, input_file in file_group.items():
            output_file = output_path / input_file.name
            process_file(input_file, output_file, original_size)

def process_batch_scenes(input_base_dir, output_base_dir, file_types=None):
    input_base_path = Path(input_base_dir)
    output_base_path = Path(output_base_dir)
    
    output_base_path.mkdir(parents=True, exist_ok=True)
    
    folders_to_process = []
    for folder in sorted(input_base_path.iterdir()):
        if folder.is_dir():
            folders_to_process.append(folder.name)
    
    for folder in folders_to_process:
        input_dir = input_base_path / folder
        output_dir = output_base_path / folder
        process_single_scene(str(input_dir), str(output_dir), file_types)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_input', type=str)
    parser.add_argument('--single_output', type=str)
    parser.add_argument('--batch_input', type=str)
    parser.add_argument('--batch_output', type=str)
    parser.add_argument('--file_types', type=str, nargs='+', 
                        help='只处理指定类型的文件，例如: --file_types png 或 --file_types jpg exr npz')
    
    args = parser.parse_args()
    
    if args.single_input and args.single_output:
        process_single_scene(args.single_input, args.single_output, args.file_types)
    elif args.batch_input and args.batch_output:
        process_batch_scenes(args.batch_input, args.batch_output, args.file_types)

