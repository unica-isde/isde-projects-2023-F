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
    """Generates and returns a color histogram of the image with the specified ID."""
    img_path = os.path.join(conf.image_folder_path, image_id)
    img = Image.open(img_path)

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Flatten the image into separate 1D arrays for each color channel
    red_channel = img_array[:, :, 0].flatten()
    green_channel = img_array[:, :, 1].flatten()
    blue_channel = img_array[:, :, 2].flatten()

    # Create a new figure
    plt.figure()

    # Generate the color histograms
    plt.hist(red_channel, bins=256, color='red', alpha=0.5, label='Red')
    plt.hist(green_channel, bins=256, color='green', alpha=0.5, label='Green')
    plt.hist(blue_channel, bins=256, color='blue', alpha=0.5, label='Blue')

    # Display the legend
    plt.legend()

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Convert the image to base64
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return image_base64