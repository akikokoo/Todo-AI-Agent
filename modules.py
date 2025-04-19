import time
from pathlib import Path
import logging as log

def log_time(func):
    """Decorator to log execution time of functions"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        log.info(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

def is_knowledge_base_empty(file_path=Path(__file__).parent / "files"):
    """Helper function to check knowledge base whether its empty or not"""
    txt_files = list(file_path.glob("*.txt"))
    if not txt_files:
        return True
    
    for file in txt_files:
        if file.stat().st_size > 0:
            return False
        
    return True
