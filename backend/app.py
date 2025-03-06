from flask import Flask, request, jsonify
import re
import time
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

app = Flask(__name__)

# Initialize Google Gemini API
genai_client = genai.Client(api_key="YOUR_GOOGLE_API_KEY")
openai_api_key = "YOUR_OPENAI_API_KEY"
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)

vector_db = None

def transcribe_video(video_path):
    """Uploads and transcribes a video."""
    video_file = genai_client.files.upload(file=video_path)
    while video_file.state.name == "PROCESSING":
        time.sleep(1)
        video_file = genai_client.files.get(name=video_file.name)
    if video_file.state.name == "FAILED":
        return None
    
    prompt = "Transcribe the video with timestamps."
    response = genai_client.models.generate_content(model="gemini-1.5-pro", contents=[video_file, prompt])
    return response.text

def create_documents(transcript, video_name):
    """Splits transcript into chunks with timestamps."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    timestamp_pattern = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]")
    lines = transcript.split("\n")
    text_chunks, timestamps = [], []
    current_timestamp, current_text = None, ""
    
    for line in lines:
        match = timestamp_pattern.search(line)
        if match:
            if current_text:
                text_chunks.append(current_text.strip())
                timestamps.append(current_timestamp)
                current_text = ""
            current_timestamp = match.group(1)
            line = line.replace(match.group(0), "").strip()
        if line.strip():
            current_text += " " + line if current_text else line
    
    if current_text:
        text_chunks.append(current_text.strip())
        timestamps.append(current_timestamp)
    
    docs = [Document(page_content=chunk, metadata={"source": video_name, "timestamp": ts}) 
            for chunk, ts in zip(text_chunks, timestamps)]
    return docs

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['file']
    transcript = transcribe_video(file.filename)
    if not transcript:
        return jsonify({"error": "Video processing failed"}), 500
    return jsonify({"transcript": transcript})

@app.route('/store', methods=['POST'])
def store_transcript():
    global vector_db
    data = request.json
    docs = create_documents(data['transcript'], data['video_name'])
    vector_db = FAISS.from_documents(docs, embedding_model)
    vector_db.save_local("faiss_index")
    return jsonify({"message": "Transcript stored successfully"})

@app.route('/query', methods=['POST'])
def query_video():
    if not vector_db:
        return jsonify({"error": "No stored transcripts found"}), 400
    data = request.json
    query = data['query']
    similar_docs = vector_db.similarity_search(query, k=3)
    return jsonify({"results": [doc.page_content for doc in similar_docs]})

if __name__ == '__main__':
    app.run(debug=True)
