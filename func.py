from fastapi import FastAPI
from typing import Literal, List
from pydantic import BaseModel
from pydantic.types import Json

app = FastAPI()

class Task(BaseModel):
    name: str
    description: str
    priority: Literal["low", "medium", "high"]
class MessageResponse(BaseModel):
    message: str
class TasksResponse(BaseModel):
    message: List[Task]

in_memory_db: List[Task] = []

@app.post("/create_task", summary= "Create a new task", operation_id="createTask")
def create_task(task_fields: Task) -> MessageResponse:
    if task_fields is None:
        return {"message": "Task not created"}
    in_memory_db.append(task_fields)
    return {"message": "Task created successfully"}

@app.get("/tasks", summary= "List all tasks", operation_id= "listTasks")
def get_tasks() -> TasksResponse:
    return {"message": in_memory_db}

@app.delete("/delete_task", summary= "Delete a task by name of the task", operation_id= "deleteTask")
def delete_task(task_name: str) -> MessageResponse:
    for index,task in enumerate(in_memory_db):
        if task.name == task_name:
            in_memory_db.pop(index)
            return {"message": "Task deleted successfully"}
    return {"message": "Task not found"}

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}