import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI
import os
from fastapi.responses import FileResponse
import json

from typing import Dict, List
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
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
from app.utils import list_images, generate_histogram
import matplotlib.pyplot as plt
import numpy as np
from app.utils import list_images

from app.forms.transformation_form import TransformationForm
from PIL import Image, ImageEnhance
import os
import time 


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
    with open('classification_scores.json', 'w') as f:
        json.dump(classification_scores, f)
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

@app.get("/histograms")
def create_histograms(request: Request):
    return templates.TemplateResponse(
        "histogram_select.html",
        {"request": request, "images": list_images()},
    )

@app.post("/histograms")
async def request_histogram(request: Request):
    form = ClassificationForm(request)
    await form.load_data()
    image_id = form.image_id

    # Generate the histogram
    histogram_base64 = generate_histogram(image_id)

    return templates.TemplateResponse(
        "histogram_output.html",
        {
            "request": request,
            "image_id": image_id,
            "histogram_base64": histogram_base64,  # Pass the histogram base64 to the context
        },
    )


@app.get("/transformations")
def create_transformation(request: Request):
    return templates.TemplateResponse(
        "transformation_select.html",
        {"request": request, "images": list_images(), "models": Configuration.models},
    )


# This function is used to delete the temporary image created after the transformation
def delete_temp_img(temp_path):
    time.sleep(1)
    os.remove("app/static/imagenet_subset/" + temp_path)


# This function is used to apply the transformation to the image and classify it
@app.post("/transformations")
async def request_transformation(request: Request, background_tasks: BackgroundTasks):
    form = TransformationForm(request)
    await form.load_data()
    image_id = form.image_id
    model_id = form.model_id

    # Open the image and apply the transformations
    with Image.open("app/static/imagenet_subset/" + image_id) as img:
        img = ImageEnhance.Color(img).enhance(form.color)
        img = ImageEnhance.Brightness(img).enhance(form.brightness)
        img = ImageEnhance.Contrast(img).enhance(form.contrast)
        img = ImageEnhance.Sharpness(img).enhance(form.sharpness)

        # Save the transformed image
        temp_path = f"temp.{image_id}"
        img.save("app/static/imagenet_subset/" + temp_path)
        img.close()

        classification_scores = classify_image(model_id = model_id, img_id = temp_path)

        # Save the classification scores to a JSON file
        with open("app/static/transformedImage.json", "w") as json_file:
            json.dump(classification_scores, json_file)
            json_file.close()

        # Render the response
        response = templates.TemplateResponse(
            "transformation_output.html",
            {
                "request": request,
                "image_id": temp_path,
                "classification_scores": json.dumps(classification_scores),
            },
        )

        # Delete the temporary image after the response is sent
        background_tasks.add_task(delete_temp_img, temp_path)
        return response

@app.post("/upload/")
async def create_upload_file(
    request: Request,
    image_id: UploadFile = File(...)
):   
    if(
        image_id.filename.endswith(".jpg") |
        image_id.filename.endswith(".JPEG") |
        image_id.filename.endswith(".png") |
        image_id.filename.endswith(".webp")
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
                "backButton" : "/delete"
            },
        )
    else:
        return templates.TemplateResponse(
           "classification_select.html",
            {
                "request": request, 
                "images": list_images(), 
                "models": Configuration.models,
                "userImage": 1
            },
        )

@app.get("/delete")
async def deleteFile(request : Request):
    if(os.path.isfile("./app/static/imagenet_subset/n00000000_usersImage.JPEG")):
        os.remove("./app/static/imagenet_subset/n00000000_usersImage.JPEG")
    return templates.TemplateResponse(
            "deleteFile.html",
            {
                "request": request,
            }
        )

@app.get("/download_scores")
async def download_scores():
    return FileResponse('classification_scores.json', media_type='application/json', filename='classification_scores.json')


@app.get("/download_plot")
async def download_plot():
    with open('classification_scores.json', 'r') as f:
        classification_scores = json.load(f)

    # Extract class labels and scores from the list
    classes, scores = zip(*classification_scores)

    # Create an index for each class
    class_indices = list(range(1, len(classes) + 1))

    # Invert the order of the classes
    class_indices = class_indices[::-1]

    # Create the appropriate dimension figure
    plt.figure(figsize=(11, len(classes)))
    # Create a bar plot
    plt.grid(alpha=0.2)
    plt.barh(class_indices, scores, color=[
        '#1A4A04', '#750014', '#795703', '#06216C', '#3F0355'])

    # Optionally rotate class labels for better readability
    plt.yticks(class_indices, classes, ha='right')
    plt.ylabel('Class')
    plt.xlabel('Score')
    plt.title('Classification Scores')

    # Save the plot as a PNG file
    plt.savefig('classification_plot.png')

    # Close the plot to free up resources
    plt.close()

    return FileResponse('classification_plot.png', media_type='application/png', filename='classification_plot.png')

