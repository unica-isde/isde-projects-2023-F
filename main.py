import json
from typing import Dict, List
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import redis
from rq import Connection, Queue
from rq.job import Job
from app.config import Configuration
from app.forms.classification_form import ClassificationForm
from app.ml.classification_utils import classify_image
from app.utils import list_images
from app.forms.transformation_form import TransformationForm
from PIL import Image, ImageEnhance
import os
import time 


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
    data = {"models": list_of_models, "images": list_of_images}
    return data


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """The home page of the service."""
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/classifications")
def create_classify(request: Request):
    return templates.TemplateResponse(
        "classification_select.html",
        {"request": request, "images": list_images(), "models": Configuration.models},
    )

@app.post("/classifications")
async def request_classification(request: Request):
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
        },
    )

@app.get("/transformations")
def create_transformation(request: Request):
    return templates.TemplateResponse(
        "transformation_select.html",
        {"request": request, "images": list_images(), "models": Configuration.models},
    )
def delete_temp_img(temp_path):
    time.sleep(1)
    os.remove("app/static/imagenet_subset/" + temp_path)

@app.post("/transformations")
async def request_transformation(request: Request, background_tasks: BackgroundTasks):
    form = TransformationForm(request)
    await form.load_data()
    image_id = form.image_id
    model_id = form.model_id

    with Image.open("app/static/imagenet_subset/" + image_id) as img:
        img = ImageEnhance.Color(img).enhance(form.color)
        img = ImageEnhance.Brightness(img).enhance(form.brightness)
        img = ImageEnhance.Contrast(img).enhance(form.contrast)
        img = ImageEnhance.Sharpness(img).enhance(form.sharpness)

        temp_path = f"temp.{image_id}"
        img.save("app/static/imagenet_subset/" + temp_path)
        img.close()

        classification_scores = classify_image(model_id = model_id, img_id = temp_path)

        with open("app/static/transformedImage.json", "w") as json_file:
            json.dump(classification_scores, json_file)
            json_file.close()

        response = templates.TemplateResponse(
            "transformation_output.html",
            {
                "request": request,
                "image_id": temp_path,
                "classification_scores": json.dumps(classification_scores),
            },
        )

        background_tasks.add_task(delete_temp_img, temp_path)
        return response