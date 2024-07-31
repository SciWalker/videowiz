import json

# Specify the path to the JSON file
file_path = 'data/outputs/output.txt'
# Open the file and load the JSON data
with open(file_path, 'r') as file:
    json_data = json.load(file)

# Access the JSON data
# Example: Print the value of a specific key
print(json_data['transcription']['results']['channels'][0]['alternatives'][0]['transcript'])
