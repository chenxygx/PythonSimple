import sys
from PIL import Image
from skimage import transform, data, io


def get_char(r, g=1, b=1, alpha=256):
    ascii_char = '''$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. '''
    if alpha == 0:
        return " "
    length = len(ascii_char)
    gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
    unit = (256.0 + 1) / length
    return ascii_char[int(gray / unit)]


def get_img(img: str):
    im = Image.open(img)
    height = int(im.size[1] / 2)
    width = int(im.size[0] / 2)

    im = im.resize((width, height), Image.NEAREST)
    txt = ""
    for h in range(height):
        for w in range(width):
            txt += get_char(im.getpixel((w, h)))
        txt += "\n"
    return txt


def convert_L(img_url):
    im = Image.open(img_url)
    im = im.convert('L')
    max_height = 400
    height = int(im.size[1] / 1.5)
    width = int(im.size[0] / 1.5)
    while True:
        if height > max_height:
            height = height / 1.5
            width = width / 1.5
        else:
            break
    im = im.resize((int(width), int(height)), Image.ANTIALIAS)
    im.save(img_url)


def main(image_name: str, art_name: str):
    convert_L(image_name)
    txt = get_img(image_name)
    with open(art_name, "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == '__main__':
    img_in = r"D:\Administrator\Desktop\1.jpg"
    img_out = r"D:\Administrator\Desktop\1.txt"
    main(img_in, img_out)
