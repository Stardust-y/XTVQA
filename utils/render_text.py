
import hashlib
import io
import os
import pdb
import random
import textwrap
from typing import Any, Callable, Iterable, List, Optional
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import jsonlines

DEFAULT_FONT_PATH = "arial.ttf"

def load_jsonl(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for line in reader:
            data.append(line)
    return data


def render_header(image: Image.Image, header: str) -> Image.Image:
  """Renders a header on a PIL image and returns a new PIL image."""
  header_image = render_text(header)
  new_width = max(header_image.width, image.width)

  new_height = int(image.height *  (new_width / image.width))
  new_header_height = int(
      header_image.height * (new_width / header_image.width))

  new_image = Image.new(
      "RGB",
      (new_width, new_height + new_header_height),
      "white")
  new_image.paste(header_image.resize((new_width, new_header_height)), (0, 0))
  new_image.paste(image.resize((new_width, new_height)), (0, new_header_height))

  return new_image



def render_text(text: str,
                text_size: int = 72,
                text_color: str = "black",
                background_color: str = "white",
                left_padding: int = 5,
                right_padding: int = 5,
                top_padding: int = 5,
                bottom_padding: int = 5,
                font_bytes: Optional[bytes] = None) -> Image.Image:
  """Render text."""
  # Add new lines so that each line is no more than 80 characters.
  wrapper = textwrap.TextWrapper(width=40)
  lines = wrapper.wrap(text=text)
  wrapped_text = "\n".join(lines)

  if font_bytes is not None:
    font_spec = io.BytesIO(font_bytes)
  else:
    font_spec = DEFAULT_FONT_PATH
  font = ImageFont.truetype(font_spec, encoding="UTF-8", size=text_size)

  # Use a temporary canvas to determine the width and height in pixels when
  # rendering the text.
  temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1), background_color))
  _, _, text_width, text_height = temp_draw.textbbox((0, 0), wrapped_text, font)

  # Create the actual image with a bit of padding around the text.
  image_width = text_width + left_padding + right_padding
  top_padding = bottom_padding = int((image_width - text_height) / 2)
  image_height = text_height + top_padding + bottom_padding
  image = Image.new("RGB", (image_width, image_height), background_color)
  draw = ImageDraw.Draw(image)
  draw.text(
      xy=(left_padding, top_padding),
      text=wrapped_text,
      fill=text_color,
      font=font)
  return image

if __name__ == "__main__":
    from dataset import load_dataset

    image = render_text("hahah")
    image
    exit()
    # for i, item in enumerate(dataset['test']):
    #     print(item['document'], item['summary'])
    #     rendered_img = render_text(item['document'])
    #     rendered_img.save(f"../../../PycharmProjects/mmm-eval/data/gigaword/test_imgs/img_{i}.png")
    #


