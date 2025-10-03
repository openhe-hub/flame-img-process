import cv2
import os
import argparse
from tqdm import tqdm

def create_video_from_images(image_folder: str, output_video_path: str, fps: int = 30):
    """
    从指定文件夹中的图片创建一个视频文件。

    Args:
        image_folder (str): 包含图片的文件夹路径。
        output_video_path (str): 输出视频文件的完整路径 (例如 'output.mp4')。
        fps (int, optional): 视频的帧率 (每秒的图片数)。默认为 30。
    """
    print("开始从图片生成视频...")
    
    # 支持的图片文件扩展名
    supported_formats = ['.png', '.jpg', '.jpeg', '.bmp']
    
    # 获取文件夹中所有支持格式的图片文件
    image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in supported_formats]
    
    # 按文件名进行排序，这是确保视频帧顺序正确的关键步骤
    image_files.sort()
    
    # 拼接成完整路径
    full_image_paths = [os.path.join(image_folder, f) for f in image_files]

    if not full_image_paths:
        print(f"错误：在文件夹 '{image_folder}' 中未找到任何支持的图片文件。")
        return

    # 读取第一张图片以获取视频的尺寸（宽度和高度）
    # 所有的图片都必须是相同的尺寸
    try:
        first_frame = cv2.imread(full_image_paths[0])
        height, width, layers = first_frame.shape
        frame_size = (width, height)
    except Exception as e:
        print(f"错误：无法读取第一张图片 '{full_image_paths[0]}'。请确保文件有效。错误信息: {e}")
        return

    print(f"检测到图片尺寸为: {width}x{height}")
    print(f"共找到 {len(full_image_paths)} 帧图片。")
    print(f"视频将以 {fps} FPS 生成。")

    # 选择视频编码器。'mp4v' 是用于 .mp4 格式的常用编码器。
    # 对于 .avi 格式，可以尝试 'XVID'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    # 创建 VideoWriter 对象
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # 使用 tqdm 创建进度条并写入每一帧
    for image_path in tqdm(full_image_paths, desc="正在生成视频帧"):
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告：跳过无法读取的文件: {image_path}")
            continue
        # 如果图片尺寸不匹配，OpenCV可能会报错或生成损坏的视频
        # 此处我们假设所有图片尺寸一致
        out.write(img)

    # 释放资源
    out.release()
    print("-" * 30)
    print(f"🎉 视频已成功生成并保存到: {output_video_path}")


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="将文件夹中的一系列图片转换为视频文件。")
    
    parser.add_argument("image_folder", type=str, 
                        help="包含输入图片的文件夹路径。")
    
    parser.add_argument("output_video", type=str, 
                        help="输出视频文件的路径和名称 (例如: ./output/my_video.mp4)。")
    
    parser.add_argument("--fps", type=int, default=30, 
                        help="视频的帧率 (Frames Per Second)。默认为 30。")

    args = parser.parse_args()

    # 调用主函数
    create_video_from_images(args.image_folder, args.output_video, args.fps)