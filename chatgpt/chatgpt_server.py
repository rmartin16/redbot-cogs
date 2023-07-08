from json import loads
from pathlib import Path
from traceback import print_exc

from flask import Flask, request, jsonify, Response

#from revChatGPT.ChatGPT import Chatbot
from revChatGPT.V1 import Chatbot

app = Flask(__name__)

CHATGPT_CONFIG_PATH = Path("/home/russell/.config/revChatGPT/config.json")
chatbot = Chatbot(config=loads(open(CHATGPT_CONFIG_PATH).read()))


@app.route('/query', methods=['POST'])
def query():
   prompt = request.json["prompt"]

   try:
      chatbot.reset_chat()
      response = ""
      print(f"Asking '{prompt}'")
      for data in chatbot.ask(prompt):
          response = data["message"]
   except Exception as e:
      print_exc()
      return jsonify({"error": f"ChatGPT ERROR {e.code}: {e.message} from {e.source} ({repr(e)})"})
   else:
      return jsonify({"answer": response})


@app.route('/reset', methods=['POST', 'GET'])
def reset():
   chatbot.reset_chat()
   return Response(status=200)


if __name__ == '__main__':
   app.run(debug=False, host="10.16.16.16")
