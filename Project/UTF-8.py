from PIL import Image, ImageDraw, ImageFont

english_text = "FIVE"
tamil_text = "ஐந்து"

img = Image.new("RGB", (600, 300), "white")
draw = ImageDraw.Draw(img)

english_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 60)
tamil_font = ImageFont.truetype("C:/Windows/Fonts/Nirmala.ttf", 60)

# Center English
draw.text((200, 60), english_text, font=english_font, fill="blue")

# Center Tamil properly below
draw.text((200, 150), tamil_text, font=tamil_font, fill="black")

img.show()
