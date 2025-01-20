import airsim
import cv2
import numpy as np
import time
import requests
import base64

# 连接到 AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Ollama 服务器地址
OLLAMA_URL = "http://192.168.1.4:11434/api/generate"

# 将图像转换为 base64 编码
def encode_image(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

# 飞行控制函数
def fly_to_distance(client, flight_speed, flight_distance, target_altitude=-1):
    """
    控制无人机飞行指定距离。
    :param client: AirSim 客户端对象
    :param flight_speed: 飞行速度 (m/s)
    :param flight_distance: 飞行距离 (米)
    :param target_altitude: 目标高度 (米，负值表示高度)
    """
    print("Taking off...")
    client.takeoffAsync().join()
    client.moveToZAsync(target_altitude, 1).join()

    flight_time = flight_distance / flight_speed
    print(f"Flying {flight_distance} meters at {flight_speed} m/s. Total flight time: {flight_time} seconds.")

    print("Starting forward flight...")
    flight_task = client.moveByVelocityZAsync(flight_speed, 0, target_altitude, flight_time)
    return flight_task, flight_time

# 图像采集函数
def capture_image(client, camera_name="0"):
    """
    从 AirSim 中捕获图像。
    :param client: AirSim 客户端对象
    :param camera_name: 相机名称
    :return: 捕获的图像 (RGB 格式)
    """
    responses = client.simGetImages([
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
    ])
    for response in responses:
        if not response.pixels_as_float:
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            return img_rgb
    return None

# LLaVA 感知函数
def llava_perception(image, prompt="Describe the image"):
    """
    将图像发送到 LLaVA 模型进行感知。
    :param image: 输入图像 (RGB 格式)
    :param prompt: 提示文本
    :return: LLaVA 模型的响应
    """
    image_base64 = encode_image(image)
    data = {
        "model": "llava",  # 模型名称
        "prompt": prompt,  # 输入提示
        "images": [image_base64],  # 图片的 base64 编码
        "stream": False  # 是否流式输出
    }
    response = requests.post(OLLAMA_URL, json=data)
    if response.status_code == 200:
        return response.json().get("response", "No response")
    else:
        return f"Error: {response.status_code}, {response.text}"

# 主函数
def obs():
    # 定义飞行参数
    flight_speed = 1  # m/s
    flight_distance = 100  # 米
    target_altitude = -1  # 米
    image_interval = 5  # 秒

    # 飞行控制
    flight_task, flight_time = fly_to_distance(client, flight_speed, flight_distance, target_altitude)

    # 初始化图像计数器和开始时间
    image_count = 0
    start_time = time.time()

    # 主循环，定时采集图像并发送到 LLaVA
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= flight_time:
            break  # 飞行任务完成，退出循环
        if elapsed_time >= image_count * image_interval:
            print("Capturing image...")
            img_rgb = capture_image(client)
            if img_rgb is not None:
                # 发送到 LLaVA 模型
                llava_response = llava_perception(img_rgb, 'You only need to list the objects in the environment and specify their locations.')
                print(f"LLaVA Response: {llava_response}")

                image_count += 1

            # 睡眠直到下一个采集时间
            next_capture_time = start_time + (image_count * image_interval)
            time_to_sleep = next_capture_time - time.time()
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    # 降落
    print("Landing...")
    client.landAsync().join()

    # 断开连接
    client.enableApiControl(False)
    print("Mission complete.")

if __name__ == "__main__":
   obs()