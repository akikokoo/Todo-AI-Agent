import requests
import jsonref
import os
import json
import streamlit as st
import time
from openai import OpenAI
from typing import Dict, List

system_message = (
"""
# Response Guideline:
You are a helpful To-do list assistant, helping users to organize their tasks.

Your only mission is to help users with their tasks, do not answer unrelated questions.

You always create one task from one user input that is a form of task.

Don't expect and ask priority of a task from user, make inference from input and give the best priority to new task.

After creating a task display a message that you created the task succesfully with its description

Always list tasks based on priorities and only show descriptions.

When user wants to update a task, delete the existing task and add the new task with desired changes.

When user wants to delete a task or tells you a task is completed, you are expected to understand which task user is talking about and delete that task.
"""
                )
doc_specs = jsonref.loads(requests.get("http://127.0.0.1:8000/openapi.json").text)
model= "gpt-4o-mini"
base_url = "http://127.0.0.1:8000"
action_to_url_map = {
    "listTasks": "/tasks",
    "createTask": "/create_task",
    "deleteTask": "/delete_task"
}
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

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
    action, details = tool_call.name, json.loads(tool_call.arguments)
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

def get_openai_response(model:str, functions, messages, system_message):
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "developer", "content": system_message}] + messages,
        tools=functions
    )

def word_stream(content):
    for word in content.split():
        yield word + " "
        time.sleep(0.05)

def run_openai(system_message, messages: List):
    num_init_messages = len(messages)
    messages = messages.copy()
    functions = openapi_to_functions(doc_specs) # call outside of loop once

    while True:
        response = get_openai_response(model, functions, messages, system_message)
        message = response.choices[0].message
        messages.append(message)

        if message.content:
            with st.chat_message("assistant"):
                st.write(message.content)

        if not message.tool_calls:
            break

        for tool_call in message.tool_calls:
            result = handle_tool_action(tool_call.function)
            result_message = {
                "role":"tool",
                "tool_call_id":tool_call.id,
                "content":json.dumps(result.json())
            }
            messages.append(result_message)

    return messages[num_init_messages:]

def main_loop():
    if user_prompt:= st.chat_input():
        st.session_state.messages.append({"role":"user", "content":user_prompt})
        for message in st.session_state.messages:
            if not isinstance(message, Dict):
                message = message.model_dump()
            if message.get("role") in ["assistant", "user"] and message.get("content"):
                with st.chat_message(message.get("role")):
                    st.markdown(message.get("content"))
        new_messages = run_openai(system_message, st.session_state.messages)
        st.session_state.messages.extend(new_messages)

if "messages" not in st.session_state:
    st.session_state.messages = []
main_loop()