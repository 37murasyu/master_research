from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np 

def putText_jp(img, text, position, font_size, color, line_width):
    # OpenCV画像をPillow形式に変換
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    # 日本語対応フォントの指定
    font = ImageFont.truetype("meiryo\meiryo.ttc", font_size)
        # line_width = 20  # 1行あたりの最大文字数

    # テキストを指定幅で改行
    wrapped_text = textwrap.fill(text, width=line_width)
    
    # テキストを描画
    draw.text(position, wrapped_text, font=font, fill=color)
    
    # Pillow形式をOpenCV形式に変換
    return np.array(img_pil)