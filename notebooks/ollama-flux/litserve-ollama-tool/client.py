# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from PIL import Image
import requests
import io
import base64
from IPython.display import display

response = requests.post("http://127.0.0.1:8000/predict", json={"prompt": "A beautiful sunset over the ocean."})
print(f"Status: {response.status_code}\nResponse:\n {response}")


# Check the status code
print(f"Status: {response.status_code}")

# Send the POST request to your FastAPI server
response = requests.post("http://127.0.0.1:8000/predict", json={"prompt": "A beautiful sunset over the ocean."})

# Check the status code
if response.status_code == 200:
    # Read the image from the response content
    image = Image.open(io.BytesIO(response.content))
    
    # Display the image directly in the notebook
    display(image)
else:
    print(f"Failed to retrieve image. Response:\n{response.text}")