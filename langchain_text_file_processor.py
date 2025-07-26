from langchain_community.document_loaders import TextLoader
import pathlib
import os
from glob import glob
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


text_dir = pathlib.Path("../../sacred_text_/")
files = glob(os.path.join(text_dir, '*.txt'))

loader = DirectoryLoader(path=text_dir, glob='*.txt', loader_cls=TextLoader)

data = loader.load()

print(data[0].metadata)

# 각 문자를 구분하여 분할

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap  = 40,
    length_function = len,
)

# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=400,
#     chunk_overlap=40,
#     encoding_name='cl100k_base'
# )

docs = text_splitter.split_documents(data)
print(len(docs))
print(docs[0].page_content)
print(docs[1].page_content)
# texts = text_splitter.split_text(data[0].page_content)
# print(len(texts))
# print(texts[0])
# for text_file in text_dir.glob("*.txt"):
#     loader = TextLoader(text_file)
#     data = loader.load()
#     print(type(data))
#     print(len(data))
#     print(data[0])
#     print(data[0].page_content)