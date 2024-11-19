from src.dataset import Anil
import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

anil = Anil(root='D:\\MyProject\llm-annotation-toolkit\datasets\\anil')
print(anil.entity)
print(anil.domain)
print(anil.category_names)
print(anil.category_descriptions)
print(anil.raw_texts)