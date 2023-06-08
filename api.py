from fastapi import FastAPI, File, UploadFile
import shutil
import uvicorn
import os
import docx2txt
import os
import PyPDF2
from pydantic import BaseModel
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
import openai
from key import *

history = []

def convert_pdf2_txt(base_path):
    files = os.listdir(base_path)
    docs = []
    for file in files:
        if file.split(".")[-1] == 'pdf':
            pdfreader = PyPDF2.PdfReader(os.path.join(base_path,file))
            texts = ""
            for i in range(len(pdfreader.pages)):
                texts+=pdfreader.pages[i].extract_text()
            docs.append(texts)
            
    return docs

def convert_docx2txt(base_path):
    docs = []
    files = os.listdir(base_path)
    for file in files:
         if file.split(".")[-1]=='docx':
              with open(os.path.join(base_path,file), 'rb') as infile:
                   doc = docx2txt.process(infile)
              docs.append(doc)

    return docs


def convert_documents(base_path,filename):
    doc1 = "\n\n".join(convert_docx2txt(base_path))
    doc2 = "\n\n".join(convert_pdf2_txt(base_path))
    
    with open(filename, 'w',encoding='utf-8') as outfile:
        outfile.write(doc1)

    with open(filename+'1','w',encoding='utf-8') as outfile:    
        outfile.write(doc2)
        
    print("done converting documents")
        

class Query(BaseModel):
    text:str


class Action:
   qa = None
   
   def load_file(self):
      embedding =  OpenAIEmbeddings()
      loader  = TextLoader('./document.txt')
      doc = loader.load()
      spliter = CharacterTextSplitter(separator='\n\n\n',chunk_size=1000,chunk_overlap=0)
      doc = spliter.split_documents(doc)
      vectorstore = Chroma.from_documents(doc, embedding)
      self.qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0,), vectorstore.as_retriever())
      self.history = []

   def knowledge_execute(self,query:str):
      
      result = self.qa({"question": query, "chat_history": self.history})['answer']
      #result = self.generate_intelligent_response(query,result)
      return result
   
   def generate_intelligent_response(self,history,query,add_info):
    
    prompt = "you are a helpful assistant. response intelligently given additional information and previous conversations,  if no information is given, answer as concisely as possible. previous_conversation: {}, question: {}, additional information: {}.".format(history,query,add_info)
    answer = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
                {"role": "system", "content": "You are an intelligent assistant. your name is omeife, your an AI developed in Africa, your build at stem focus in Nigeria, you like to chat with humans and also you like to learn more about african culture and people, your name means on that does wonder, one that does great things"},
                {"role": "user", "content": prompt},
            ],
        temperature=0,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.0
    )
    final_answer= answer['choices'][0]['message']['content']
    
    return final_answer
   
    
   def generate_history(self,history,curr_quest,system_answer):
        
        text = "User: {}, System: {}".format(curr_quest,system_answer)
        if len(history)>=5:
            history.pop(0)
            history.append(text)
        else:
            history.append(text)
            
        return history
   
   def gpt_bot(self,question):

    global history
    added_info = self.knowledge_execute(question)

    if len(history)<1: 
        answer = self.generate_intelligent_response("No previous history",question,added_info)
    else:
        answer = self.generate_intelligent_response(" ,".join(history),question,added_info)
        
    history = self.generate_history(history,question,answer)
    
    
    return answer
    



app = FastAPI()
engine = Action()

@app.post("/uploader/")
def create_upload_file(file: UploadFile = File(...)):
   print("file: ",file.file)
   try:
    with open("testing_document", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    engine.load_file()
    return {'filename':file.filename}
   except Exception as e:
      return {"filename": str(e)}
   
@app.post('/query/')
def get_response(q:Query):
   text = q.text
   if engine.qa:
      resp = engine.gpt_bot(text)
      return {"text": resp}
   
   else:
      return 'upload a document first...'

if __name__ == "__main__":
   uvicorn.run(app,port=8080)