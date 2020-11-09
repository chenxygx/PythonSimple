from captcha.image import ImageCaptcha
import random
import numpy as np
import pandas as pd
import string

# 一、生成随机验证码
# 设置字符集
characters = string.digits + string.ascii_uppercase
# print(characters)

# 设置图片长宽、字符个数和字符集类别数量
width, height, n_len, n_class = 170, 80, 4, len(characters)


# # 定义生成器
# generator = ImageCaptcha(width=width, height=height)

# trans_x = []
# trans_y = []
# for i in range(1):
#     # 随机生成4个字符
#     random_str = ''.join([random.choice(characters) for j in range(n_len)])
#
#     # 把字符转变成图片
#     img = generator.generate_image(random_str)
#     img_name = f'./images/{i}.jpg'
#     img.save(img_name)
#     trans_x.append(np.asarray(img))
#     trans_y.append(random_str)
#
# np.savez('data.npz', labels=trans_y, images=trans_x)
#
# load = np.load('data.npz')
# labels = load['labels']
# images = load['images']
# images = images.astype(np.float)
# images = np.multiply(images, 1.0 / 255.0)
# print(images)

# 定义数据生成器
def gen123():
    batch_size = 32
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            # 生产随机字符
            random_str = ''.join([random.choice(characters) for j in range(4)])
            # 生产图片
            X[i] = generator.generate_image(random_str)
            # 把Y设置成4个列表
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


if __name__ == '__main__':
    for n in gen123():
        print(n)
