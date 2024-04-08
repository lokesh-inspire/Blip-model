from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import datetime

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


st = datetime.datetime.now()
raw_image = Image.open('show.jpg').convert('RGB')

inputs = processor(raw_image, return_tensors="pt")

out = model.generate(**inputs,max_new_tokens=50)
print(processor.decode(out[0], skip_special_tokens=True))
et = datetime.datetime.now()

print("\nTime taken : ", (et-st).total_seconds())

