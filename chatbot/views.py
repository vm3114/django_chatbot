from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import auth
from django.contrib.auth.models import User
from .models import Chat
from django.utils import timezone
# Create your views here.

#algorithm starts here
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import fitz
import nltk
import spacy
import google.generativeai as genai
from dotenv import load_dotenv
from IPython.display import Markdown
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pdfminer.high_level import extract_text

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.vectorstores import Chroma
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
import re
vector_index = None
all_pages = ""

def setup_environment():
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=api_key)
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key,
                                   temperature=0.2, convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    return model, embeddings

def preprocess_text(text):
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    nlp = spacy.load("en_core_web_sm")
    stop_words = set(stopwords.words('english'))
    
    doc = nlp(text)
    processed_words = [token.lemma_ for token in doc if token.text.lower() not in stop_words]
    
    return ' '.join(processed_words).strip()

def extract_text_from_pdf(pdf_path):
    text = ""
    document = fitz.open(pdf_path)
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return preprocess_text(text)

def load_pdfs(pdf_folder_path):
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
    all_pages = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder_path, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        all_pages.append(text)
    
    return all_pages

def initialize_data():
    global vector_index, all_pages
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder_path = os.path.join(base_dir, "pdf_files")

    model, embeddings = setup_environment()
    
    all_pages = load_pdfs(pdf_folder_path)
    context_all = "\n\n".join(all_pages).strip()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=750)
    texts = text_splitter.split_text(context_all)
    
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})

chat_history = []

def update_chat_history(question, answer):
    global chat_history
    chat_history.extend([HumanMessage(content=question), answer])

def answer_query(question, chat_history):
    model, embeddings = setup_environment()
    
    instruction_to_system = """
    Given a chat history and the latest user question
    which might reference context in the chat history formulate a standalone question
    which can be understood without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is.
    Do not use bold/italic or bullet points anywhere. If you need to use bullet points you can just put "-" for a bullet
    """
    
    question_maker_prompt = ChatPromptTemplate.from_messages(
        [("system", instruction_to_system),
         MessagesPlaceholder(variable_name="chat_history"),
         ("human", "{question}")]
    )
    
    question_chain = question_maker_prompt | model | StrOutputParser()
    
    qa_system_prompt = """You are an online mental health therapist.
    Use the following pieces of retrieved context to answer the question.
    You are free to add on additional information/instructions/advice for the user.
    First try to calm them if they are anxious/depressed, then proceed to answer their query/give them suitable feedback.
    
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", qa_system_prompt),
         MessagesPlaceholder(variable_name="chat_history"),
         ("human", "{question}")]
    )
    
    def contextualize_question(input: dict):
        if input.get("chat_history"):
            return question_chain
        else:
            return input["question"]
    
    retriever_chain = RunnablePassthrough.assign(
        context=contextualize_question | vector_index
    )
    
    rag_chain = (
        retriever_chain
        | qa_prompt
        | model
        | StrOutputParser()
    )
    
    ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
    update_chat_history(question, ai_msg)
    
    return ai_msg

# # Example usage
# question = "Why am I depressed?"
# chat_history = []
# answer = answer_query(question, chat_history)
# print(answer)


#algorithm ends here

def chatbot(request):
    chats = Chat.objects.filter(user = request.user) if request.user.is_authenticated else None
    if request.method == 'POST':
        message = request.POST.get('message')
        response = answer_query(message, chat_history)

        if request.user.is_authenticated:
            chat = Chat(user = request.user, message = message, response = response, time = timezone.now())
            chat.save()

        return JsonResponse({'message':message, 'response': response})
    return render(request, 'chatbot.html', {'chats': chats})


def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)

        if user is not None:
            auth.login(request, user)
            return redirect('chatbot')
        else:
            error_msg = "Invalid username or password"
            return render(request, 'login.html', {'error_message': error_msg})

    else:
        return render(request, 'login.html')

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 == password2:
            try:
                user = User.objects.create_user(username, email, password1)
                user.save
                auth.login(request, user)
                return redirect('chatbot')
            
            except:
                error_msg = 'Error creating account'
                return render(request, 'register.html', {'error_message': error_msg})

        else:
            error_msg = "Passwords don't match"
            return render(request, 'register.html', {'error_message': error_msg})

    return render(request, 'register.html')

def logout(request):
    auth.logout(request)
    return redirect('login')