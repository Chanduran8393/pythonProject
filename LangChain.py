# Below Libraries to be installed
#pip install pypdf
#pip install faiss-cpu #to store content & memory FAISS Facebook AI Similarity Search
#pip install langchain # Search & flow library
#pip install openai
#pip install tiktoken
# For Reference https://platform.openai.com/docs/guides/embeddings

import os

os.environ['OPENAI_API_KEY'] = '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'

#export OPENAI_API_KEY="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"

#Step-1

# to make the Text Chunks here and print the result
from langchain.document_loaders import PyPDFLoader
loader=PyPDFLoader("D:\Chanduran\LEARNING\Python\AstraZeneca_AR_2022.pdf") ##Change the Location here
pages_content=loader.load_and_split()
print(len(pages_content),pages_content)

#Step-2
#Using Embeddings to convert Text Chunks into Vectors which is Numeric values

from langchain.embeddings.openai import OpenAIEmbeddings  #OpenAIEmbeddings is a Class
from langchain.vectorstores import FAISS #FAISS is also a Class

openai = OpenAIEmbeddings(openai_api_key="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")

#initiate the Class here
embeddings=OpenAIEmbeddings()
db=FAISS.from_documents(pages_content,embeddings)

# to Save in FAISS
db.save_local("faiss_index")

# to Load

new_db=FAISS.load_local("faiss_index",embeddings)

## Getting Below Error

#raise self.handle_error_response(
#openai.error.RateLimitError: You exceeded your current quota, please check your plan and billing details.
