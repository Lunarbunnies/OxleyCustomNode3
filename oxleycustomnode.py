import sys
import os

# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the external_packages directory
external_packages_path = os.path.join(current_script_dir, 'external_packages')

# Add this path to sys.path
sys.path.append(external_packages_path)

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import torch  # Import torch
import websocket
import json
from json.decoder import JSONDecodeError
import base64

class OxleyWebsocketDownloadImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"ws_url": ("STRING", {})},  # WebSocket URL to connect to
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    FUNCTION = "download_image_ws"
    CATEGORY = "oxley"

    def download_image_ws(self, ws_url):
        # Initialize the WebSocket client and connect to the server
        ws = websocket.create_connection(ws_url)

        # Receive a message
        message = ws.recv()
        ws.close()  # Close the connection once the message is received

        try:
            # Attempt to parse the message as JSON
            data = json.loads(message)
        except JSONDecodeError:
            # Handle cases where the message is not valid JSON
            print(f"Received non-JSON message: {message}")
            return None
    
        if "image" in data:
            # Process the message assuming it contains an 'image' field encoded in Base64
            try:
                # Decode the Base64 image data
                image_data = base64.b64decode(data["image"].split(",")[1])
                image = Image.open(BytesIO(image_data))
            except Exception as e:
                # Handle potential errors in decoding or opening the image
                print(f"Error processing image data: {e}")
                return None
        else:
            # Handle cases where the expected 'image' field is not found in the JSON
            print("No image data found in the received message")
            return None

        # Convert the image to RGB format
        image = image.convert("RGB")

        # Convert the image to a NumPy array and normalize it
        image_array = np.array(image).astype(np.float32) / 255.0

        # Convert the NumPy array to a PyTorch tensor
        image_tensor = torch.from_numpy(image_array)

        # Add a new batch dimension at the beginning
        image_tensor = image_tensor[None,]

        # Return the PyTorch tensor with the batch dimension added
        return (image_tensor,)

    @classmethod
    def IS_CHANGED(cls, ws_url):
        # Logic to determine if the node should re-execute, potentially based on WebSocket URL changes
        from datetime import datetime
        return datetime.now().isoformat()

class OxleyWebsocketPushImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_tensor": ("TENSOR", {}),  # Input image tensor
                "ws_url": ("STRING", {}),  # WebSocket URL to push the image to
            },
        }

    RETURN_TYPES = ("STRING",)  # Possible return type for confirmation/message
    RETURN_NAMES = ("status_message",)
    FUNCTION = "push_image_ws"
    CATEGORY = "oxley"

    def tensor_to_pil(self, image_tensor):
        """
        Convert a PyTorch tensor to a PIL Image.
        """
        # Assuming the tensor is in CxHxW format and in the 0-1 range
        image_tensor = image_tensor.squeeze()  # Remove batch dimension if present
        image_tensor = image_tensor.mul(255).byte()  # Convert to 0-255 range
        image = Image.fromarray(image_tensor.cpu().numpy(), 'RGB')  # Convert to PIL Image
        return image

    def push_image_ws(self, image_tensor, ws_url):
        # Convert tensor to PIL Image
        image = self.tensor_to_pil(image_tensor)
        
        # Convert PIL Image to JPEG bytes
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        jpeg_bytes = buffer.getvalue()
        
        # Encode JPEG bytes to Base64
        base64_bytes = base64.b64encode(jpeg_bytes)
        base64_string = base64_bytes.decode('utf-8')
        
        # Initialize WebSocket client and connect to the server
        ws = websocket.create_connection(ws_url)
        
        # Prepare the message (You might want to wrap this in JSON or directly send the Base64 string)
        message = json.dumps({"image": base64_string})
        
        # Send the message
        ws.send(message)
        ws.close()  # Close the connection after sending the message
        
        return ("Image sent successfully",)

class OxleyWebsocketReceiveJsonNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ws_url": ("STRING", {}),  # WebSocket URL to connect to
                "fields": ("LIST_STRING", {})  # List of fields to extract from the JSON
            },
        }

    RETURN_TYPES = ("DICT_STRING",)  # Assuming output as string might be sufficient
    RETURN_NAMES = ("json_out",)
    FUNCTION = "receive_json_ws"
    CATEGORY = "oxley"

    def receive_json_ws(self, ws_url, fields):
        # Initialize the WebSocket client and connect to the server
        ws = websocket.create_connection(ws_url)

        # Receive a message
        message = ws.recv()
        ws.close()  # Close the connection once the message is received

        try:
            # Attempt to parse the message as JSON
            data = json.loads(message)
        except JSONDecodeError:
            # Handle cases where the message is not valid JSON
            print(f"Received non-JSON message: {message}")
            return None

        # Initialize an empty dictionary for the output
        output_data = {}

        # Extract specified fields from the JSON data
        for field in fields:
            if field in data:
                # Add the field value to the output dictionary, convert to string if necessary
                output_data[field] = str(data[field])
            else:
                print(f"Field '{field}' not found in the received message")

        # Return the extracted data
        return (output_data,)

    @classmethod
    def IS_CHANGED(cls, ws_url, fields):
        # Logic to determine if the node should re-execute, potentially based on WebSocket URL changes or fields list changes
        from datetime import datetime
        return datetime.now().isoformat()


class OxleyCustomNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "image_in" : ("IMAGE", {}) },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    FUNCTION = "invert"
    CATEGORY = "oxley"

    def invert(self, image_in):
        image_out = 1 - image_in
        return (image_out,)

class OxleyDownloadImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "url" : ("STRING", {}) },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)

    FUNCTION = "download_image"
    CATEGORY = "oxley"

    def download_image(self, url):
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Open the image using Pillow
        image = Image.open(BytesIO(response.content))
        
        # Convert the image to RGB format
        image = image.convert("RGB")

        # Convert the image to a NumPy array and normalize it
        image_array = np.array(image).astype(np.float32) / 255.0

        # Convert the NumPy array to a PyTorch tensor
        image_tensor = torch.from_numpy(image_array)

        # Add a new batch dimension at the beginning
        image_tensor = image_tensor[None,]

        # Return the PyTorch tensor with the batch dimension added
        return (image_tensor,)

    @classmethod
    def IS_CHANGED(cls, url):
        # Always returns a unique value to force the node to be re-executed, e.g. returning a timestamp
        from datetime import datetime
        return datetime.now().isoformat()
