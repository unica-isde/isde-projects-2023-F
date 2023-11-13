import json
import os
from typing import Dict, List, Annotated
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import redis
import shutil
from rq import Connection, Queue
from rq.job import Job
from app.config import Configuration
from app.forms.classification_form import ClassificationForm
from app.ml.classification_utils import classify_image
from app.utils import list_images
from starlette.datastructures import URL


app = FastAPI()
config = Configuration()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/info")
def info() -> Dict[str, List[str]]:
    """Returns a dictionary with the list of models and
    the list of available image files."""
    list_of_images = list_images()
    list_of_models = Configuration.models
    data = {
        "models": list_of_models, 
        "images": list_of_images
        }
    return data


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """The home page of the service."""
    return templates.TemplateResponse(
        "home.html", 
        {
            "request": request
            }
        )    

@app.get("/classifications")
def create_classify(request: Request):
    return templates.TemplateResponse(
        "classification_select.html",
        {
            "request": request, 
            "images": list_images(), 
            "models": Configuration.models, 
            "userImage": 0
            },
    )

@app.post("/classifications")
async def request_classification(
        request: Request
    ):
    form = ClassificationForm(request)
    await form.load_data()
    image_id = form.image_id     
    model_id = form.model_id       
    classification_scores = classify_image(model_id=model_id, img_id=image_id)
    return templates.TemplateResponse(
        "classification_output.html",
        {
            "request": request,
            "image_id": image_id,
            "classification_scores": json.dumps(classification_scores),            
            "backButton" : "/classifications"
        },        
    )
    
@app.get("/users_image")
def create_classify(request: Request):
    return templates.TemplateResponse(
        "classification_select.html",
        {
            "request": request, 
            "images": list_images(), 
            "models": Configuration.models,
            "userImage": 1
        },
    )

@app.post("/users_image")
async def users_image(request: Request):
    form = ClassificationForm(request)
    await form.load_data()
    image_id = "n00000000_usersImage.JPEG"
    model_id = form.model_id
    classification_scores = classify_image(model_id=model_id, img_id=image_id)   
    return templates.TemplateResponse(
        "classification_output.html",
        {
            "request": request,
            "image_id": image_id,
            "classification_scores": json.dumps(classification_scores),
            "backButton" : "/users_image"
        },
    )

@app.post("/upload/")
async def create_upload_file(
    request: Request,
    image_id: UploadFile = File(...)
):   
    with open("./app/static/imagenet_subset/n00000000_usersImage.JPEG", "wb") as buffer:
        shutil.copyfileobj(image_id.file, buffer)
        form = ClassificationForm(request)      
    await form.load_data()
    image_id = "n00000000_usersImage.JPEG"
    model_id = form.model_id
    classification_scores = classify_image(model_id=model_id, img_id=image_id)
    request._url = URL("/classifications")
    return templates.TemplateResponse(
        "classification_output.html",
        {
            "request": request,
            "image_id": image_id,
            "classification_scores": json.dumps(classification_scores),
            "backButton" : "/users_image"
        },
    )