import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image, HarmCategory, HarmBlockThreshold
import google.auth as auth
from google.cloud import storage
from google.cloud import bigquery
from google.api_core import exceptions as gcs_exceptions

from PIL import Image, ImageDraw, ImageColor, ImageFont
import cv2

import io
import os
import requests
from io import BytesIO

import json
import random
import numpy as np
import tempfile
import datetime

from IPython.display import display, HTML

# DECLARATIONS
billing_project = "placeholder-project-id"
gcp_location = "us-central1"
model_name = "gemini-2.5-flash"
# embedding_model_name = "multimodalembedding@001" ## not in use

# GCS Configurations
# * Raw Folder: Image of found luggages before segmentation
# * Processed Folder: Image of individual found luggage after segmentation
# * Search Folder: Image of lost luggage
gcs_bucket_name = "bucket-name"
gcs_source_folder = "raw"
gcs_processed_folder = "processed"
gcs_search_folder = "search"
storage_client = None

# BigQuery Configurations
bq_table_id = "project-id.dataset-id.table-id"
bigquery_client = None

# Modified System Instructions for BigQuery
bounding_box_system_instructions = """
    Return bounding boxes as a JSON array. For each object, include:
    1. "label": A short name for the object (e.g., "red suitcase").
    2. "box_2d": The bounding box coordinates.
    Never return masks or code fencing. Limit to 25 objects.
    """

# Function: Authenticate Project
def authenticate_project():
    global billing_project
    try:
        credentials, project_id = auth.default()
        if not project_id:
            project_id = credentials.quota_project_id

        print(f"✅ | Authentication successful!")
        print(f"📦 | project_id : {project_id}")
        billing_project = project_id

    except auth.exceptions.DefaultCredentialsError:
        print("❌ Authentication failed. ")
        print("Please configure Application Default Credentials by running:")
        print("gcloud auth application-default login")

# Initialize Clients
def init_clients():
    global client, storage_client, bigquery_client
    try:
        client = vertexai.init(project=billing_project, location=gcp_location)
        print(f"✅ | GenAI Client initialized successfully!")
    except Exception as e:
        print(f"❌ GenAI Client authentication failed. {e}")

    try:
        storage_client = storage.Client()
        print(f"✅ | GCS Client initialized successfully!")
    except Exception as e:
        print(f"❌ GCS Client initialization failed. {e}")

    try:
        bigquery_client = bigquery.Client()
        print(f"✅ | BigQuery Client initialized successfully!")
    except Exception as e:
        print(f"❌ BigQuery Client initialization failed. {e}")

# Helper Functions
# PARSE JSONS
def parse_json(json_output: str):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if "```json" in line:
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output

# A helper function that helps with rendering images from URL in colab results
def get_image_tag(gcs_uri):
    try:
        return f'<img src=\"{gcs_uri}\" width=\"150\" />'
    except Exception as e:
        return "Image not found"

# Drawing Utility
additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def plot_bounding_boxes(im, bounding_boxes, source_image_name):
    img = im.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)

    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray', 'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal', 'olive', 'coral', 'lavender', 'violet', 'gold', 'silver'] + additional_colors
    bounding_boxes = parse_json(bounding_boxes)
    font = ImageFont.load_default()

    folder_name_from_source = os.path.splitext(source_image_name)[0]
    crop_counter = 1

    # --- BQ LOGGING: Prepare list for found objects ---
    objects_found_list = []

    for i, bounding_box in enumerate(json.loads(bounding_boxes)):
        color = colors[i % len(colors)]
        abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
        abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
        abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
        abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)
        if abs_x1 > abs_x2: abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2: abs_y1, abs_y2 = abs_y2, abs_y1

        try:
            cropped_image = im.crop((abs_x1, abs_y1, abs_x2, abs_y2))
            crop_buffer = io.BytesIO()
            cropped_image.save(crop_buffer, format="JPEG")
            crop_buffer.seek(0)
            output_filename = f"{folder_name_from_source}-output-{crop_counter:04d}.jpg"
            crop_output_blob_name = f"{gcs_processed_folder}/{output_filename}"
            bucket = storage_client.bucket(gcs_bucket_name)
            crop_blob = bucket.blob(crop_output_blob_name)
            crop_blob.upload_from_file(crop_buffer, content_type="image/jpeg")
            print(f"✅ Cropped object {crop_counter} uploaded to GCS: gs://{gcs_bucket_name}/{crop_output_blob_name}")

            # --- BQ LOGGING: Create a record for this object ---
            object_record = {
                "output_uri": f"gs://{gcs_bucket_name}/{crop_output_blob_name}",
                "item_color": bounding_box.get("color"),
                "item_gen_desc": bounding_box.get("general_description"),
                "item_det_desc": bounding_box.get("detailed_description")
            }
            objects_found_list.append(object_record)

            crop_counter += 1
        except Exception as e:
            print(f"❌ Error cropping or uploading object {crop_counter}: {e}")

        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)
        if "label" in bounding_box:
            imageNumber = crop_counter - 1
            drawInText = f"{imageNumber} {bounding_box['label']}"
            #draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)
            draw.text((abs_x1 + 8, abs_y1 + 6), drawInText, fill=color, font=font)

    print(f"✅ Image annotation complete. Attempting to upload main image to GCS...")
    try:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        bucket = storage_client.bucket(gcs_bucket_name)
        main_output_blob_name = f"processed-overview/processed_{source_image_name}"
        blob = bucket.blob(main_output_blob_name)
        blob.upload_from_file(buffer, content_type="image/jpeg")
        main_output_uri = f"gs://{gcs_bucket_name}/{main_output_blob_name}"
        print(f"✅ Main image uploaded to GCS: {main_output_uri}")

    except Exception as e:
        print(f"❌ Error uploading main image to GCS. {e}")

# Detecting 2D Bounds
def detect_2d_bound_box(image_name, user_prompt):
    global client
    try:
        print(f"Downloading {image_name} from GCS...")
        bucket = storage_client.bucket(gcs_bucket_name)
        source_blob_name = f"{gcs_source_folder}/{image_name}"
        blob = bucket.blob(source_blob_name)
        image_bytes = blob.download_as_bytes()

        im = Image.open(BytesIO(image_bytes))

        print("Sending image to Gemini API for analysis...")
        image_part = Part.from_data(image_bytes, mime_type="image/jpeg")
        model = GenerativeModel(
            model_name,
            system_instruction=[bounding_box_system_instructions]
        )
        response = model.generate_content(
            [user_prompt, image_part],
            generation_config={"temperature": 0.5}
        )
        print(f"✅ Detecting 2D Bound Box success.")
        plot_bounding_boxes(im, response.text, image_name)

    except gcs_exceptions.NotFound:
        print(f"❌ Error: File not found in GCS at gs://{gcs_bucket_name}/{source_blob_name}")
    except Exception as e:
        print(f"❌ An error occurred during image processing: {e}")

# Perform Image Analysis
def image_analysis():
  file_name="file-name.jpg"
  user_prompt="Detect bags and luggages only."
  detect_2d_bound_box(file_name, user_prompt)

# Running the workflow
authenticate_project()
init_clients()
image_analysis()