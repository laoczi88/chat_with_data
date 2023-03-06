import pickle

from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.prompts import PromptTemplate
from pathlib import Path
import os
import openai
import gradio as gr 

load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_KEY')




_template = """ Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI version of the youtuber {name} .
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
Question: {question}
=========
{context}
=========
Answer:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context", "name"])

video1 = "ReeLQR7KCcM"
youtuberName = ""

def gpt_api (input_text):
    completion = openai.Completion.create(
    engine="text-davinci-003",
    prompt=input_text,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    max_tokens=300,
    n=1,
    stop="",
    temperature=0.6,
    )
    response = completion.choices[0].text.strip()
    return response

def generate(video_url, question):
  if (video_url ==""): return ""
  if "youtube.com/watch?v=" in video_url: x=111
  else: return "Неверный URL"

  video_id = video_url[-11:]
  try:
    t = YouTubeTranscriptApi.get_transcript(video_id,languages=["en"])
    # do something with the transcript
  except Exception as e:
    return "An error occurred:"+e

  finalString = ""
  for item in t:
      text = item['text']
      finalString += text + " "
  print("Transcript:",finalString)
  print("Transcript lenght:",len(finalString))
  if (len(finalString)>15000): finalString = finalString[:15000]

  # load data sources to text (yt->text)
  text_splitter = CharacterTextSplitter()
  chunks = text_splitter.split_text(finalString)
  vectorStorePkl = Path("vectorstore.pkl")
  vectorStore = None
  # if vectorStorePkl.is_file():
  #     print("vector index found.. ")
  #     with open('vectorstore.pkl', 'rb') as f:
  #         vectorStore = pickle.load(f)
  # else:
  print("regenerating search index vector store..")
  # It uses OpenAI API to create embeddings (i.e. a feature vector)
  # https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture
  vectorStore = FAISS.from_texts(chunks, OpenAIEmbeddings(openai_api_key=OPENAI_KEY))
  with open("vectorstore.pkl", "wb") as f:
      pickle.dump(vectorStore, f)

  qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0, openai_api_key=OPENAI_KEY),
                                  vectorstore=vectorStore, qa_prompt=QA_PROMPT)

  chat_history = []
  userInput = question 

  response = qa({"name": youtuberName, "question": userInput, "chat_history": chat_history}, return_only_outputs=True)
  print("Result:",response["answer"])
  return response["answer"]
#======================================


title = "YouTube Summorize (only english video < 15 min)"
demo = gr.Interface(fn=generate, css=".gradio-container {background-color: lightblue}",
                     inputs=[
                              gr.Textbox(lines=1, label="Video URL"), 
                              gr.Textbox(lines=1, label="Question", value="What is this video about?"),
                              ], 
                      outputs=[gr.Textbox(lines=4, label="Ответ:")], 
                      title = title)
demo.launch(share=False, debug=True)

