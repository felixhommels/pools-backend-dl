import requests

def main():
    image_path = "test1.png"
    lat = 47.312734
    zoom = 20

    url = "http://localhost:8000/predict/"

    with open(image_path, "rb") as f:
        files = {"file": ("test1.png", f, "image/png")}
        data = {"lat": str(lat), "zoom": str(zoom)}

        response = requests.post(url, files=files, data=data)
        
    if response.ok:
        print("✅", response.json())
    else:
        print("❌ Error:", response.status_code)
        print(response.text)   

    
if __name__ == "__main__":
    main()