import google.generativeai as genai
import os
# API_KEY = "AIzaSyB2rGDZzkVKxgkV8y_uJf4LvK9E9WKfWoE"
API_KEY = "AIzaSyCYo6MWJKX4nrV8i36GKVVEVeuYfD3co-s"
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content('Please summarise this document: ...')

print(response.text)