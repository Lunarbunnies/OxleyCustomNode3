import requests
from io import BytesIO
from PIL import Image
import numpy as np
import torch  # Import torch

import asyncio
import websockets
import json

class OxleyWebsocketDownloadImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "ws_url" : ("STRING", {}) },  # WebSocket URL to connect to
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    FUNCTION = "download_image_ws"
    CATEGORY = "oxley"

    async def download_image_async(self, ws_url):
        async with websockets.connect(ws_url) as websocket:
            message = await websocket.recv()
            data = json.loads(message)
            if "image" in data:
                image_data = base64.b64decode(data["image"].split(",")[1])
                return Image.open(BytesIO(image_data))
            else:
                raise ValueError("No image data found in the received message")

    def download_image_ws(self, ws_url):
        # Run the asynchronous download_image_async function and wait for it to complete
        loop = asyncio.get_event_loop()
        image = loop.run_until_complete(self.download_image_async(ws_url))
        
        # Process the image similar to OxleyDownloadImageNode
        image = image.convert("RGB")
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array)
        image_tensor = image_tensor[None,]  # Add batch dimension
        
        return (image_tensor,)

    @classmethod
    def IS_CHANGED(cls, ws_url):
        # Implement appropriate logic to determine if the node should re-execute, e.g., based on WebSocket URL changes
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
