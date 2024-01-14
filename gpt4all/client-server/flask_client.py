import requests
import json

# The URL of the Flask server's predict endpoint
server_url = 'http://localhost:4892/predict'
#server_url = 'http://localhost:4891/predict'

# The prompt to send to the server
prompt = "Who is Michael Jordan?"

# Prepare the data to send in the POST request
data = {
    'prompt': prompt
}

# Send the POST request to the server
response = requests.post(server_url, json=data)

# Check if the request was successful
if response.status_code == 200:
    # Print the response from the server
    print("Server response:", response.json())
else:
    print("Failed to get response from server, status code:", response.status_code)
