from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY"))

docs = [Document(page_content="hi"), Document(page_content = "hello")]
docs2 = [Document(page_content="bye")]
safe_video_name = "docs"
safe_video_name_list_docs = [f"{safe_video_name}_{i}" for i in range(len(docs))]
safe_video_name2 = "docs2"
safe_video_name_list_docs2 = [f"{safe_video_name2}_{i}" for i in range(len(docs2))]
vector_db_doc = FAISS.from_documents(docs, embedding_model, ids = safe_video_name_list_docs)
vector_db_doc2 = FAISS.from_documents(docs2, embedding_model, ids = safe_video_name_list_docs2)

vector_db_doc.merge_from(vector_db_doc2)
document_ids = list(vector_db_doc.docstore._dict.keys())

print(vector_db_doc.get_by_ids(["docs2_0"]))
print("=====================================")
print(document_ids)

