import requests
from PIL import Image
import io

class FluxTool:
    description = 'Generates an image based on a text prompt and returns the result.'
    parameters = {
        'type': 'object',
        'properties': {
            'prompt': {
                'type': 'string',
                'description': 'The text prompt to generate the image from.',
            },
        },
        'required': ['prompt'],
    }

    def __init__(self):
        self.server_url = "http://127.0.0.1:8000/predict"
        self.default_prompt = "a cute kitty"
        self.prompt = None
        self.response = None

    def send_request(self, prompt=None):
        """Send a request to the server with the given prompt."""
        if prompt is None:
            prompt = self.default_prompt
        
        self.prompt = prompt
        self.response = requests.post(self.server_url, json={"prompt": self.prompt})
        return self.response.status_code
    
    def query_flux(self, prompt: str):
        """Handles the full process of sending a request and processing the response."""
        status = self.send_request(prompt)  # Send the prompt to the server and get status
        return self.process_response(), status  # Process the response and return the image and status
    
    def process_response(self):
        """Process the server response, returning the image if successful."""
        if self.response.status_code == 200:
            image = Image.open(io.BytesIO(self.response.content))
            return image
        else:
            print(f"Failed to retrieve image. Response:\n{self.response.text}")
            return None

    async def execute(self, prompt: str):
        image, status = self.query_flux(prompt)
        return {
            'prompt': prompt,
            'status': status,
            'response': image
        }
