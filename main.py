from fastapi import FastAPI
from app import *

app = FastAPI()

@app.get("/")
async def root():
    
    # Define example request
    request = {"URL": "https://www.gutenberg.org/cache/epub/64317/pg64317.txt", "author": "F. Scott Fitzgerald", 
               "title": "The Great Gatsby"}
    
    # Call function and pass in request
    result = book_parser(request)
    
    # Note that FastAPI automatically serializes and converts a dictionary
    # to JSON format 
        
    return result