import os
from PIL import Image, ImageDraw, ImageFont
import random

TEMPLATE = './test/res/Image__2018-03-15__14-57-29.bmp'
LENGTH = 13
CHARS=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
tm = Image.open(TEMPLATE)
font = ImageFont.truetype("SEMI_OCR_Font_document.ttf", 28)
# im = Image.new('RGB', (900, 50), (255, 255, 255))
xy = (226, 255)
label = open('./test/training/label.txt', 'a')
for i in range(200):
  im = tm.copy()
  draw = ImageDraw.Draw(im)
  code = random.sample(CHARS, LENGTH)
  code = ''.join(code)
  label.write(code+'\n')
  color = random.randrange(0, 128)
  x = random.randrange(223, 229)
  y = random.randrange(253, 257)
  draw.text((x,y), code, font=font, fill=color)
  im = im.resize((400, 300), Image.BICUBIC)
  im.save('./test/training/{0}.png'.format(i))

label.close()




