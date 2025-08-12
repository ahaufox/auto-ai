import requests
import base64
import os

base_url = 'http://127.0.0.1:8007/'


def test_probe():
    """测试probe接口"""
    print("Testing probe endpoint...")
    r = requests.get(f"{base_url}probe/")
    print(f"Response status: {r.status_code}")
    print(f"Response content: {r.content}")
    return r.status_code == 200


def test_parse():
    """测试parse接口"""
    print("\nTesting parse endpoint...")
    
    # 读取图像文件并转换为base64
    image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "imgs", "input", "saved_image.png")
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            print(f"Image loaded successfully: {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return False
    
    # 准备请求数据
    payload = {"base64_image": base64_image}
    
    # 发送POST请求
    try:
        r = requests.post(f"{base_url}parse/", json=payload)
        print(f"Response status: {r.status_code}")
        print(f"Response content: {r.text[:500]}...")  # 只打印部分响应，避免输出过长
        
        # 如果响应成功，可以进一步处理返回的结果
        if r.status_code == 200:
            result = r.json()
            print(result.get('parsed_content_list'))
            print(f"Cost: {result.get('latency')} seconds")
            # 可以将返回的图像保存到文件
            if 'som_image_base64' in result:
                output_image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "imgs", "out", "test_result.png")
                with open(output_image_path, "wb") as f:
                    f.write(base64.b64decode(result['som_image_base64']))
                print(f"Result image saved to: {output_image_path}")
        return r.status_code == 200
    except Exception as e:
        print(f"Error sending request: {e}")
        return False


if __name__ == "__main__":
    # probe_success = test_probe()
    parse_success = test_parse()