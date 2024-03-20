import matplotlib.pyplot as plt
import os
from fastapi.responses import FileResponse, HTMLResponse
import json
from typing import Dict, List
from fastapi import FastAPI, Request, File, UploadFile, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
from app.config import Configuration
from app.forms.classification_form import ClassificationForm
from app.ml.classification_utils import classify_image
from app.utils import list_images, generate_histogram
from app.forms.transformation_form import TransformationForm
from PIL import Image, ImageEnhance
import time
from starlette.datastructures import URL
import random

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

    unique_id = str(random.randint(1, 100000))
    path = f"app/scores/classification_scores{unique_id}.json"
    with open(path, 'w') as f:
        json.dump(classification_scores, f)
    return templates.TemplateResponse(
        "classification_output.html",
        {
            "request": request,
            "image_id": image_id,
            "unique_id": unique_id,
            "classification_scores": json.dumps(classification_scores),
            "backButton": "/classifications"
        },
    )


@app.get("/users_image")
def create_classify(request: Request):
    """
    Endpoint to render the classification_select.html template and pass necessary data.

    Args:
        request (Request): The HTTP request object.

    Returns:
        TemplateResponse: The rendered template response.
    """
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
    """
    Handle the POST request for user's image classification, and gives it a placeholder name: n00000000_usersImage.JPEG.

    Args:
        request (Request): The incoming request object.

    Returns:
        TemplateResponse: The template response containing the classification output.
    """
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
            "backButton": "/users_image"
        },
    )


@app.post("/upload/")
async def create_upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    image_id: UploadFile = File(...)
):
    """
    Create an upload file.
    Checks whether the file is an image.
    If this is the case, gives it a random name.

    Args:
        request (Request): The request object.
        background_tasks (BackgroundTasks): The background tasks object.
        image_id (UploadFile): The uploaded image file.

    Returns:
        TemplateResponse: The response containing the classification output or the classification selection page.
    """
    rand = random.randint(1, 10000000)

    try:
        temp_name = f"n{rand}_usersImage.JPEG"
        temp_path = f"./app/static/imagenet_subset/{temp_name}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image_id.file, buffer)
            form = ClassificationForm(request)
        await form.load_data()
        image_id = temp_name
        model_id = form.model_id
        classification_scores = classify_image(
            model_id=model_id, img_id=image_id)
        request._url = URL("/classifications")
        response = templates.TemplateResponse(
            "classification_output.html",
            {
                "request": request,
                "image_id": image_id,
                "classification_scores": json.dumps(classification_scores),
            },
        )
        background_tasks.add_task(delete_temp_img, temp_path)
        
        return response
    except:
        response= templates.TemplateResponse(
            "classification_select.html",
            {
                "request": request,
                "images": list_images(),
                "models": Configuration.models,
                "userImage": 1
            },
        )

        background_tasks.add_task(delete_temp_img, temp_path)
        return response


@app.get("/histograms")
def create_histograms(request: Request):
    """
    Endpoint to create histograms.

    Args:
        request (Request): The HTTP request object.

    Returns:
        TemplateResponse: The HTML template response containing the image selection page.
    """
    return templates.TemplateResponse(
        "histogram_select.html",
        {"request": request, "images": list_images()},
    )


@app.post("/histograms")
async def request_histogram(request: Request):
    """
    Handle the POST request for generating histograms.

    Args:
        request (Request): The incoming request object.

    Returns:
        TemplateResponse: The response containing the histogram of the selected image.
    """
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
            # Pass the histogram base64 to the context
            "histogram_base64": histogram_base64,
        },
    )


@app.get("/transformations")
def create_transformation(request: Request):
    """
    Handle GET request for the '/transformations' endpoint.
    
    Args:
        request (Request): The incoming request object.
    
    Returns:
        TemplateResponse: The template response containing the 'transformation_select.html' template,
        along with the request object and the list of images and models.
    """
    return templates.TemplateResponse(
        "transformation_select.html",
        {"request": request, "images": list_images(), "models": Configuration.models},
    )


# This function is used to delete the temporary image created after the transformation

def delete_temp_img(temp_path):
    """
    Deletes the temporary image file at the specified path.

    Args:
        temp_path (str): The path of the temporary image file to be deleted.

    Returns:
        None
    """
    time.sleep(1)
    os.remove(temp_path)


# This function is used to apply the transformation to the image and classify it
@app.post("/transformations")
async def request_transformation(request: Request, background_tasks: BackgroundTasks):
    """
    Handle the POST request for image transformations.

    Args:
        request (Request): The incoming request object.
        background_tasks (BackgroundTasks): The background task, used to delete the temporary image after a set amount of time.

    Returns:
        TemplateResponse: The response containing the transformed image and classification scores.
    """
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
        temp_name = f"temp.{image_id}"
        temp_path = f"app/static/imagenet_subset/{temp_name}"
        img.save(temp_path)
        img.close()

        classification_scores = classify_image(model_id=model_id, img_id=temp_name)

        # Render the response
        response = templates.TemplateResponse(
            "transformation_output.html",
            {
                "request": request,
                "image_id": temp_name,
                "classification_scores": json.dumps(classification_scores),
            },
        )

        # Delete the temporary image after the response is sent
        background_tasks.add_task(delete_temp_img, temp_path)
        return response


@app.get("/download_scores")
async def download_scores(request: Request, background_tasks: BackgroundTasks):
    """
    Download scores endpoint.

    This endpoint allows users to download classification scores in JSON format.

    Parameters:
    - request: The incoming request object.
    - background_tasks: Background tasks to be executed.

    Returns:
    - FileResponse: The file response containing the classification scores in JSON format.
    """
    unique_id = request.query_params.get("unique_id")
    temp_path = "app/scores/classification_scores"+unique_id+".json"
    return FileResponse(temp_path, media_type='application/json', filename='classification_scores.json')


@app.get("/download_plot")
async def download_plot(request: Request, background_tasks: BackgroundTasks,):
    """
    Generate and download a classification plot.

    Args:
        request (Request): The HTTP request object.
        background_tasks (BackgroundTasks): The background tasks object.

    Returns:
        FileResponse: The file response containing the generated plot.
    """
    unique_id = request.query_params.get("unique_id")
    with open("app/scores/classification_scores"+unique_id+".json", 'r') as f:
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
    plt.savefig(
        'app/scores/classification_plot'+unique_id+'.png')

    # Close the plot to free up resources
    plt.close()

    temp_path = "app/scores/classification_plot"+unique_id+".png"
    return FileResponse(temp_path, media_type='application/png', filename='classification_plot.png')


@app.post("/delete-content")
async def delete_content(request: Request):
    """
    Delete content based on the provided unique_id.

    Args:
        request (Request): The HTTP request object.

    Returns:
        dict: A dictionary containing the success message.
    """
    unique_id = request.query_params.get("unique_id")
    if os.path.exists("app/scores/classification_plot"+unique_id+".png"):
        os.remove("app/scores/classification_plot"+unique_id+".png")
    if os.path.exists("app/scores/classification_scores"+unique_id+".json"):
        os.remove("app/scores/classification_scores"+unique_id+".json")
    return {"message": "Content deleted successfully"}
