from flask import Flask, request, jsonify, render_template
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

app = Flask(__name__)

# Load the PDF file and extract text
file_path = '400 Questions & Technicals.pdf'
pdf = fitz.open(file_path)
text = ''
for page in pdf:
    text += page.get_text()

# Load a pre-trained sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline('text-generation', model='gpt2', pad_token_id=50256)

@app.route('/') 
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data['question']
    
    # Retrieve relevant context
    context = retrieve_context(text, question, model)
    
    # Generate a response using the local model
    answer = generate_response(context, question)

    return jsonify({'context': context, 'answer': answer})

# Function to retrieve relevant context based on a query
def retrieve_context(text, question, model):
    sentences = text.split('. ')
    question_embedding = model.encode(question, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # Compute similarity scores
    cosine_scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]
    best_sentence_idx = cosine_scores.argmax().item()

    # Retrieve the best-matching sentence
    context = sentences[best_sentence_idx]
    
    # Remove the question from the context if it's repeated
    if question.lower() in context.lower():
        context = context.replace(question, '')

    return context


# Function to generate a response using a local language model
def generate_response(context, question):
    prompt = f"Based on the following context, answer the question:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    
    # Using a local model for text generation
    generator = pipeline('text-generation', model='gpt2', pad_token_id=50256)  # Specify pad_token_id
    
    # Increase the max_length for the response
    response = generator(prompt, max_length=500, num_return_sequences=1, truncation=False)  # Set max_length to a higher value
    return response[0]['generated_text'].split('Answer:')[1].strip()


if __name__ == '__main__':
    app.run(debug=True)

#add strenghth of transformers 