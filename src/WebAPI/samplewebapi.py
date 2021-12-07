import requests
response  = requests.post('http://35.75.180.35/api/behavior', json={"id": "0", "behavior": "2"})

print("Status code: ", response.status_code)
print("Printing Entire Post Request")
print(response.json())
