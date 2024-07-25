import logging
from flask import Flask, request, jsonify,Response
from deep_translator import GoogleTranslator
from langchain_community.document_loaders import PyMuPDFLoader
from flask_cors import CORS  
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from threading import Lock
import hashlib

# Load environment variables
load_dotenv()

# Initialize Flask app and configure CORS
app = Flask(__name__)
CORS(app)

# Configure logging
log_file_path = 'app.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)

# Load configuration from environment variables
openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER')
app.config['EMBEDDING_FOLDER'] = os.getenv('EMBEDDING_FOLDER')

# Global variables
lock = Lock()
pdf_processing_complete = {}
embedding_cache = {}
ALLOWED_EXTENSIONS = {'pdf'}

# Check if upload and embedding directories exist, create if not
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.mkdir(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['EMBEDDING_FOLDER']):
    os.mkdir(app.config['EMBEDDING_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_pdf_hash(pdf_file_path):
    """Calculate SHA-256 hash of the PDF content."""
    hash_sha256 = hashlib.sha256()
    with open(pdf_file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def extract_text_from_pdf(pdf_file):
    loader = PyMuPDFLoader(pdf_file)
    data = loader.load_and_split()
    return data

def handle_empty_pdf(data, pdf_file_path):
    if len(data) == 0:
        os.remove(pdf_file_path)
        return False, jsonify({'message': 'Please upload a true PDF (not scanned pdf file).'})
    return True, None

def save_embeddings(embeddings_path, data):
    db = FAISS.from_documents(data, OpenAIEmbeddings())
    db.save_local(embeddings_path)

def check_processing_and_inputs(pdf_processing_complete, user_id, user_question):
    if user_id not in pdf_processing_complete or not pdf_processing_complete[user_id]:
        return jsonify({'response': 'Document processing is not complete. Please process the Document first.'}), 400
    if not user_id:
        return jsonify({'response': 'User ID is missing.'}), 400
    if not user_question:
        return jsonify({'response': 'Please provide a question in the question form field.'}), 400
    return None

def create_chat_model():
    return ChatOpenAI(
    temperature=0,
    max_tokens=1000,
    model = os.getenv("MODEL_NAME"),
    api_key = os.getenv("OPENAI_API_KEY"),
    streaming=True
    )

def create_embeddings():
    return OpenAIEmbeddings()

def create_prompt_template():
    template = """
    You are a helpful assistant. Read and observe the PDF content carefully, understand the user's query,think step by step , and provide a detailed and accurate answer based on the PDF context.
    If user ask question which not related to PDF context then give user a warning message.Always generate response in a proper format.

    # Context: {context}
    # Question: {query}
    # Answer: """
    return PromptTemplate(input_variables=["Answers", "query"], template=template)

def create_llm_chain():
    return LLMChain(llm=create_chat_model(), prompt=create_prompt_template(), verbose=True)

def load_db(user_id):
    return FAISS.load_local(f"embeddings/{user_id}", create_embeddings(),allow_dangerous_deserialization=True)

def fetch_docs(db, query):
    return db.similarity_search(query, k=2)

def generate_response(query, docs):
    context = " ".join([doc.page_content for doc in docs])
    return create_llm_chain().predict(context=context, query=query)

def translate_response(response, target_language):
    return GoogleTranslator(source='auto', target=target_language).translate(response)

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    global pdf_processing_complete
    with lock:
        try:
            pdf_file = request.files.get('pdf_file')
            if not pdf_file or pdf_file.filename == '':
                logger.warning('No PDF file provided in the upload file section.')
                return jsonify({'message': 'Please provide a PDF file in the upload file section.'}), 400
            
            if not allowed_file(pdf_file.filename):
                logger.warning('Invalid file type provided. Only PDF files are allowed.')
                return jsonify({'message': 'Only PDF files are allowed.'}), 400
            
            pdf_file_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
            pdf_file.save(pdf_file_path)
            logger.info(f'PDF file saved at {pdf_file_path}')
            
            user_id_hash = calculate_pdf_hash(pdf_file_path)
            embeddings_path = os.path.join(app.config['EMBEDDING_FOLDER'], user_id_hash)
            
            if os.path.exists(embeddings_path):
                pdf_processing_complete[user_id_hash] = True
                os.remove(pdf_file_path)
                logger.info(f'Document already processed. Returning user ID: {user_id_hash}')
                return jsonify({'message': 'Document processed successfully', 'user_id': user_id_hash}), 200
            
            data = extract_text_from_pdf(pdf_file_path)
            pdf_processing_complete[user_id_hash], response = handle_empty_pdf(data, pdf_file_path)
            if not pdf_processing_complete[user_id_hash]:
                logger.warning('Empty PDF uploaded, responding with error.')
                return response, 400
            
            if not os.path.exists(embeddings_path):
                save_embeddings(embeddings_path, data)
                logger.info(f'Embeddings saved at {embeddings_path}')
                
            pdf_processing_complete[user_id_hash] = True
            os.remove(pdf_file_path)
            logger.info(f'Document processed successfully. User ID: {user_id_hash}')

            return jsonify({'message': 'Document processed successfully', 'user_id': user_id_hash}), 200
        except Exception as e:
            logger.error(f'Error processing PDF: {str(e)}', exc_info=True)
            return jsonify({'error': str(e)}), 500
        
@app.route('/ask_question', methods=['POST'])
def ask_question():
    global pdf_processing_complete
    with lock:
        query = request.form.get('question', '')
        user_id = request.form.get('user_id', '')
        language = request.form.get('language', '').lower()

        response = check_processing_and_inputs(pdf_processing_complete, user_id, query)
        if response:
            logger.warning('Input validation failed when asking question.')
            return response

        retriever = load_db(user_id).as_retriever()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
                {"context": retriever | format_docs, "query": RunnablePassthrough()}
                | create_prompt_template()
                | create_chat_model()
                | StrOutputParser()
            )
        # def generate():
        #     llm_response = rag_chain.stream(query)
        #     for chunk in llm_response:
        #         yield chunk
        
        if not language:
            language = 'english'
            # translated_response = llm_response
            def generate():
                llm_response = rag_chain.stream(query)
                for chunk in llm_response:
                    yield chunk
            Response(generate(), mimetype='text/event-stream')
        else:
            llm_response = rag_chain.invoke(query)
            translated_response = translate_response(llm_response, language)
            return jsonify({"response":translated_response})

        # logger.info(f'Question asked by user {user_id}: {query}')
        # logger.info(f'Response generated: {response_text}')

        # return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=False,threaded=True) 
