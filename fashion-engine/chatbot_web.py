#!/usr/bin/env python3
"""
Simple Flask web interface for Fashion Chatbot with images
"""
from flask import Flask, render_template_string, request, jsonify
from src.chatbot import FashionChatbot

app = Flask(__name__)
chatbot = FashionChatbot()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fashion Advisor Chat</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .chat-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 700px;
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .chat-header h1 { font-size: 1.5rem; margin-bottom: 5px; }
        .chat-header p { opacity: 0.8; font-size: 0.9rem; }
        .chat-messages {
            height: 500px;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 12px;
            white-space: pre-wrap;
            line-height: 1.5;
        }
        .message.user {
            background: #667eea;
            color: white;
            margin-left: 20%;
        }
        .message.bot {
            background: white;
            color: #333;
            margin-right: 20%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .message.loading {
            background: #e9ecef;
            color: #666;
            font-style: italic;
        }
        .outfit-item {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .outfit-item img {
            width: 80px;
            height: 80px;
            object-fit: cover;
            border-radius: 8px;
        }
        .outfit-item .info {
            flex: 1;
        }
        .outfit-item .name {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .outfit-item .color {
            font-size: 0.85rem;
            color: #666;
        }
        .chat-input {
            display: flex;
            padding: 15px;
            background: white;
            border-top: 1px solid #eee;
        }
        .chat-input input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.3s;
        }
        .chat-input input:focus {
            border-color: #667eea;
        }
        .chat-input button {
            margin-left: 10px;
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .chat-input button:hover { transform: scale(1.05); }
        .chat-input button:disabled { opacity: 0.6; cursor: not-allowed; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>👗 Fashion Advisor</h1>
            <p>Get personalized outfit recommendations with photos</p>
        </div>
        <div class="chat-messages" id="messages">
            <div class="message bot">👋 Hello! Ask me anything like:<br>• "What should I wear for a casual spring day?"<br>• "Recommend an outfit for a date tonight"<br>• "Generate a trendy winter outfit"</div>
        </div>
        <div class="chat-input">
            <input type="text" id="userInput" placeholder="Ask for outfit recommendations..." autofocus>
            <button id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const messagesDiv = document.getElementById('messages');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');

        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        function addMessage(text, isUser) {
            const div = document.createElement('div');
            div.className = 'message ' + (isUser ? 'user' : 'bot');
            div.innerHTML = text;
            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return;

            addMessage(text, true);
            userInput.value = '';
            sendBtn.disabled = true;

            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message loading';
            loadingDiv.textContent = 'Thinking...';
            messagesDiv.appendChild(loadingDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: text})
                });
                const data = await res.json();
                loadingDiv.remove();
                
                // Format response with images
                let formatted = data.response;
                // Convert Image: http... to img tags
                formatted = formatted.replace(/Image: (https:\/\/images\.unsplash\.com[^\\n]+)/g, 
                    '<br><img src="$1" style="width:120px;height:120px;object-fit:cover;border-radius:8px;margin-top:5px;">');
                
                addMessage(formatted, false);
            } catch (err) {
                loadingDiv.remove();
                addMessage('Error: ' + err.message, false);
            }

            sendBtn.disabled = false;
            userInput.focus();
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    response = chatbot.chat(message)
    return jsonify({'response': response})

if __name__ == '__main__':
    print("=" * 50)
    print("   FASHION CHATBOT WEB INTERFACE")
    print("=" * 50)
    print("\n🚀 Server running at: http://localhost:5000")
    print("   Press Ctrl+C to stop\n")
    app.run(port=5000, debug=True)
