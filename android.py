import subprocess
import time


# 查看系统版本
# pull_cmd = ["adb", "shell", 'getprop', 'ro.build.description']
# subprocess.run(pull_cmd, check=True)


def take_screenshot(device_id=None):
    # 定义截图文件在设备上的路径
    remote_path = "/data/local/tmp/screenshot.png"

    # 定义截图文件在本地计算机上的路径
    local_path = "screenshot.png"
    time.sleep(1)
    # 构造adb命令
    if device_id:
        screencap_cmd = ["adb", "-s", device_id, "shell", "screencap", "-p", remote_path]
        pull_cmd = ["adb", "-s", device_id, "pull", remote_path, local_path]
    else:
        screencap_cmd = ["adb", "shell", "screencap", "-p", remote_path]
        pull_cmd = ["adb", "pull", remote_path, local_path]
    # adb shell getprop ro.build.version.release
    # 执行截图命令
    subprocess.run(screencap_cmd, check=True)
    time.sleep(1)

    # 将截图拉取到本地
    subprocess.run(pull_cmd, check=True)


def wack_up(device_id=None):
    if device_id:
        wake_up_cmd = ["adb", "-s", device_id, "shell", "input", "keyevent", "26"]
    else:
        wake_up_cmd = ["adb", "shell", "input", "keyevent", "26"]
    subprocess.run(wake_up_cmd, check=True)
    time.sleep(1)


wack_up()

take_screenshot()
