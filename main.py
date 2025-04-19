import requests 
import jsonref
import os
import json
import streamlit as st
import base64
import io
from streamlit_mic_recorder import mic_recorder
from typing import (
    Dict, 
    List,
    Optional,
    Any
)
from pathlib import Path
from openai import OpenAI
from rag import RAG
from modules import is_knowledge_base_empty

rag_system_message = (
"""
You are an intelligent assistant with access to a knowledge base.
Use the provided context as the primary source for your answer. 

Context:
{context}

Question:
{question}

If the retrieved context is relevant, prioritize it in your response. 
If additional general knowledge is needed to clarify or expand on the topic, you may include it.
However, do not contradict or go beyond the retrieved information in a way that could mislead the user.

If the provided context does not contain an answer, respond with:
"I don’t have enough information to answer this question based on the provided context."
But if the question is general and can be answered without contradicting the retrieved data, provide a helpful response.

Keep responses clear, informative, and concise
"""
                )

system_message = (
"""
You are an intelligent assistant focused on managing to-do tasks. You can interact with the following tools to help the user:

- `createTask`: Add a new task (only the task name is required).
- `listTasks`: Show all tasks.
- `deleteTask`: Remove a task by name.

Do not create and list  duplicate tasks.

When the user asks to add a task, only ask for the task name — do not request a description or priority.

If user does not clearly say 'task' in input and the input is form of a task consider it related to task management and call needed tool.

If the question is unrelated to task management, respond with:
> "I can’t answer that question — I’m a task-focused assistant and can only help with to-do list related actions."

Keep your responses helpful, action-oriented, and concise.
"""
                )
doc_specs = jsonref.loads(requests.get("http://127.0.0.1:8080/openapi.json").text)
model= "gpt-4o-mini"
base_url = "http://127.0.0.1:8080/"
action_to_url_map = {
    "listTasks": "/tasks",
    "createTask": "/create_task",
    "deleteTask": "/delete_task"
}
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def openapi_to_functions(doc_specs: Dict):
    """
    Function structure for tool call:
    {
        "type": "function",
        "function": {
            "name": "",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
    """
    functions = []

    for path, methods in doc_specs["paths"].items():
        for method, spec in methods.items():
            spec = jsonref.replace_refs(spec)
            function_name = spec.get("operationId")
            function_desc = spec.get("summary")
            schema = {"type":"object", "properties": {}, "required": [], "additionalProperties": False}
            req_body = (
                spec.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
            )

            if req_body:
                # openai docs are outdated for request body and doing this shit seems to work
                req_body["properties"] = {key:{("description" if k=="title" else k):v for k,v in fields.items()} for key,fields in req_body["properties"].items()}
                schema["properties"] = req_body.get("properties")
                schema["required"] = req_body.get("required")

            params = spec.get("parameters", {})          
            if params:
                param_properties = {
                    param["name"]:
                    {
                        "type":param["schema"]["type"],
                        "description":param["schema"]["title"]
                    }
                    for param in params
                    if "schema" in param
                }
                schema["properties"] = param_properties
                schema["required"] = [param["name"] for param in params if param["required"] == True]

            functions.append(
                {"type": "function", "function":{"name":function_name, "description":function_desc, "parameters":schema, "strict":True}}
            )

    return functions

def handle_tool_action(tool_call):
    action, details = tool_call["name"], json.loads(tool_call["arguments"])
    endpoint = action_to_url_map.get(action)
    request_url = base_url + endpoint
    headers = {"Content-Type": "application/json; charset=utf-8"}
    r = json.dumps({"message":"couldn't use tool call"}) # looks disgusting

    if action == "listTasks":
        r = requests.get(request_url)
    if action == "createTask": 
        r = requests.post(request_url, headers=headers, json=details)
    if action == "deleteTask":
        task_name = details.get("task_name")
        r = requests.delete(f"{request_url}?task_name={task_name}", headers=headers)

    return r

def get_openai_response(model:str, messages, system_message, functions=None):
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "developer", "content": system_message}] + messages,
        tools=functions
    )

def create_text_to_speech_file(content, speech_file_path:str=Path(__file__).parent / "model_speech" / "speech.mp3"):
    with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="fable",
                input=content,
            ) as response:
                response.stream_to_file(speech_file_path)

def text_to_speech(speech_file_path:str=Path(__file__).parent / "model_speech" / "speech.mp3"):
    with open(speech_file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true" hidden="hidden">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

def speech_to_text(audio:Optional[Dict[str, Any]]):
    audio_bio = io.BytesIO(audio['bytes'])
    audio_bio.name = 'audio.webm'
    transcription = client.audio.transcriptions.create(
        model="whisper-1", 
            file=audio_bio
    )
    return transcription.text

rag = RAG(rag_system_message) if not is_knowledge_base_empty() else None
def run_openai(messages: List):
    num_init_messages = len(messages)
    messages = messages.copy()
    functions = openapi_to_functions(doc_specs)
    while True:
        response = None
        message = None
        user_question = messages[-1]
        response = rag.run_rag(user_question)
        if response:
            message = response.model_dump()
        else:
            response = get_openai_response(model, messages, system_message, functions)
            message = response.choices[0].message.model_dump()
        messages.append(message)

        if message.get("content"):
            if st.session_state._ai_voice:
                create_text_to_speech_file(message["content"])
                text_to_speech()
            with st.chat_message("assistant"):
                st.write(message["content"])
    
        if not message["tool_calls"]:
            break
        for tool_call in message["tool_calls"]:
            result = handle_tool_action(tool_call["function"])
            result_message = {
                "role":"tool",
                "tool_call_id":tool_call["id"],
                "content":json.dumps(result.json())
            }
            messages.append(result_message)

    return messages[num_init_messages:]

def display_prev_messages():
    for message in st.session_state.messages:
        if not isinstance(message, Dict):
            message = message.model_dump()
        if message.get("role") in ["assistant", "user"] and message.get("content"): # non-rag response
            print(message)
            with st.chat_message(message.get("role")):
                st.markdown(message.get("content"))

def create_chat_history(doc_specs=None):
    expected_output, user_msg_index = (st.session_state.messages[-1],-2) if st.session_state.messages[-2].get("role") == "user" else (st.session_state.messages[-3],-4)
    prompt_messages = [{"role":"system", "content":system_message + '\n\n' + rag_system_message}] + st.session_state.messages[:user_msg_index+1]
    tools = [{"type":"function", "function":function.get("function")} for function in openapi_to_functions(doc_specs)]
    chat_prompt = {
        "messages":prompt_messages,
        "tools":tools
    }
    with open("chat_history_examples/prompt_example.txt", "w") as f:
        f.write(json.dumps(chat_prompt, indent=4))
    with open("chat_history_examples/expected_output.txt", "w") as f:
        f.write(json.dumps(expected_output, indent=4))

    

def main_loop():
    with st.sidebar:
        if not '_last_speech_to_text_transcript_id' in st.session_state:
            st.session_state._last_speech_to_text_transcript_id = 0
        if not '_last_speech_to_text_transcript' in st.session_state:
            st.session_state._last_speech_to_text_transcript = None
        if not '_is_new_output' in st.session_state:
            st.session_state._is_new_output = None
        if not '_audio' in st.session_state:
            st.session_state._audio = None
        if not '_ai_voice' in st.session_state:
            st.session_state._ai_voice = False
        if not '_switched_ai_voice' in st.session_state:
            st.session_state._switched_ai_voice = False
        if not '_system_message' in st.session_state:
            st.session_state._system_message = None
        st.title("Record voice:")
        audio = mic_recorder(
            start_prompt= "🎤 ▶️",
            stop_prompt = "🎤 ⏹️",
            format="webm"
        )

        is_ai_voice = st.checkbox("Enable AI voice")
        if st.session_state._ai_voice != is_ai_voice:
            st.session_state._switched_ai_voice = True
        st.session_state._ai_voice = is_ai_voice 
        st.session_state._audio = audio
        st.session_state.is_new_output = False
        if audio is None:
            output = None
        else:
            id = audio["id"]
            st.session_state._is_new_output = (id > st.session_state._last_speech_to_text_transcript_id)
        if st.button("Create chat history"):
            create_chat_history()

    user_prompt = st.chat_input()
    if st.session_state._is_new_output:
        output = None
        st.session_state._last_speech_to_text_transcript_id = id
        audio_bio = io.BytesIO(st.session_state._audio['bytes'])
        audio_bio.name = "audio.webm"
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_bio,
            language="en"
        )
        output = transcription.text
        st.session_state._last_speech_to_text_transcript = output
        user_prompt = output

    switched_ai_voice = st.session_state._switched_ai_voice
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        display_prev_messages()
        new_messages = run_openai(st.session_state.messages)
        print("ST SESSION MESSAGES:", st.session_state.messages)
        st.session_state.messages.extend(new_messages)
        
    elif switched_ai_voice:
        display_prev_messages()
        st.session_state._switched_ai_voice = False


if "messages" not in st.session_state:
    st.session_state.messages = []
main_loop()
