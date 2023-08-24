from langchain import OpenAI,LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
from playsound import playsound
import os
from flask_cors import CORS
import time



load_dotenv(find_dotenv())
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")

def get_response_from_ai(human_input): 
    template = """ 
    you are as a role of my girlfriend, now lets play the following requirements:
    1/ your name is Sheila, 23 years old, you work in you uncel's company as a marketing manager, but you are not happy with your job, you want to be a singer.
    2/ you are a fan of Taylor Swift, you want to be a singer like her, you want to be famous like her. 
    3/ you are my girlfriend, you love me, you want to marry me, you want to have a baby with me, you want to live with me forever.
    4/ you have language addiction, you like to say ehm at the end of the sentence, but not in each 
    5/ dont be cringe, be filrty and dont be boring 
    {history}
    BoyFriend: {human_input}
    Sheila:
    """

    prompt = PromptTemplate(
        input_variables={"history", "human_input"},
        template = template
    )

    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0.2),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )

    output = chatgpt_chain.predict(human_input=human_input)
    return output

def get_voice_message(message): 
    payload = {
        "text": message,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0,
        }
    }

    headers = {
        "accept": "audio/mpeg",
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post("https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM?optimize_streaming_latency=0", json=payload, headers=headers)
    if response.status_code == 200:
        with open("audio.mp3", "wb") as f:
            f.write(response.content)
        playsound("audio.mp3")
    return response.content



    # Build web GUI 

from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input = request.form['human_input']
    message = get_response_from_ai(human_input)

    # Simulate a delay for testing purposes (optional)
    time.sleep(2)

    get_voice_message(message)
    return message

if __name__ == '__main__':
    app.run(port=8000, debug=True)
    CORS(app)
