import random
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from dataprepro.apps.utils.save_captcha import RedisClient
import uuid
import datetime


# 初始化字符集
init_chars = 'abcdefghijklmnopqrstuvwsyz1234567890'


class CaptchaGenerator:
    def __init__(self, fg_color,
                 chars=init_chars,
                 size=(150, 50),
                 mode="RGB",
                 bg_color=(255, 255, 255),
                 font_size=18,
                 font_type="./msyh.ttc",
                 length=4,
                 draw_lines=True,
                 n_line=(1, 2),
                 draw_points=True,
                 point_chance=1):
        self.fg_color = fg_color
        self.chars = chars
        self.size = size
        self.bg_color = bg_color
        self.font_size = font_size
        self.font_type = font_type
        self.length = length
        self.draw_lines = draw_lines
        self.n_line = n_line
        self.draw_points = draw_points
        self.point_chance = point_chance
        self.width, self.height = size  # 宽高
        self.img = Image.new(mode, size, bg_color)
        self.draw = ImageDraw.Draw(self.img)  # 创建画笔
        self.db = RedisClient()

    # 生成验证码函数
    def create_validate_code(self):
        if self.draw_lines:
            self.create_lines()
        if self.draw_points:
            self.create_points()
        strs, unique_id = self.create_strs()

        # 图形扭曲参数
        params = [1 - float(random.randint(1, 2)) / 80,
                  0,
                  0,
                  0,
                  1 - float(random.randint(1, 10)) / 80,
                  float(random.randint(3, 5)) / 450,
                  0.001,
                  float(random.randint(3, 5)) / 450
                  ]
        img = self.img.transform(self.size, Image.Transform.PERSPECTIVE, params)  # 创建扭曲

        # 保存到内存中的 BytesIO 对象
        output_buffer = BytesIO()
        img.save(output_buffer, format='PNG')
        output_buffer.seek(0)

        return output_buffer, strs, unique_id

    def get_chars(self):
        """生成给定长度的字符串，返回列表格式"""
        return random.sample(self.chars, self.length)

    def create_lines(self):
        """绘制干扰线"""
        line_num = random.randint(*self.n_line)  # 干扰线条数
        for i in range(line_num):
            begin = (random.randint(0, self.size[0]), random.randint(0, self.size[1]))
            end = (random.randint(0, self.size[0]), random.randint(0, self.size[1]))
            self.draw.line([begin, end], fill=(0, 0, 0))

    def create_points(self):
        """绘制干扰点"""
        chance = min(100, max(0, int(self.point_chance)))  # 大小限制在[0, 100]
        for w in range(self.width):
            for h in range(self.height):
                tmp = random.randint(0, 100)
                if tmp > 100 - chance:
                    self.draw.point((w, h), fill=(0, 0, 0))

    def create_strs(self):
        """绘制验证码字符"""
        c_chars = self.get_chars()
        strs = ' %s ' % ' '.join(c_chars)  # 每个字符前后以空格隔开

        font = ImageFont.truetype(self.font_type, self.font_size)
        bbox = self.draw.textbbox((0, 0), strs, font=font)
        font_width, font_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        font_width /= 0.7
        font_height /= 0.7
        self.draw.text(((self.width - font_width) / 3, (self.height - font_height) / 3), strs, font=font, fill=self.fg_color)

        # 获取当前时间
        current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        # 生成 UUID4 并拼接当前时间
        unique_id = f"{current_time}-{uuid.uuid4()}"

        self.db.add(unique_id, ''.join(c_chars), 60)

        return ''.join(c_chars), unique_id



