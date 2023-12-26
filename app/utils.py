import os

from app.config import Configuration
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from fastapi.responses import StreamingResponse
import io

conf = Configuration()


def list_images():
    """Returns the list of available images."""
    img_names = filter(
        lambda x: x.endswith(".JPEG"), os.listdir(conf.image_folder_path)
    )
    return list(img_names)

def generate_histogram(image_id):
    """Generates and returns a grayscale histogram of the image with the specified ID."""
    img_path = os.path.join(conf.image_folder_path, image_id)
    img = Image.open(img_path)

    # Convert the image to grayscale
    img_gray = img.convert('L')

    # Convert the grayscale image to a numpy array
    img_array = np.array(img_gray)

    # Flatten the image into a 1D array
    flattened_img = img_array.flatten()

    # Create a new figure
    plt.figure()

    # Generate the histogram
    plt.hist(flattened_img, bins=256, color='gray', alpha=0.7)

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Convert the image to base64
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return image_base64