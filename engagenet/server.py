import aiohttp
import asyncio
import base64

async def send_image(image_path):
    # URL of your Docker server
    url = "http://localhost:9001/overhead-angle-detection-6lmpn/3?api_key=R5i9d6qtGJCDn0LiaEhe"
    
    # Read the image and encode in base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())

    # Headers
    headers = {
        "Content-Type": "application/json"
    }

    # Send the request using aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=encoded_image, headers=headers) as response:
            print(await response.text())

# List of image paths
image_paths = ["./data/test.png", "./data/test2.png", "./data/test3.png"]

# Run the asynchronous function for each image
async def main():
    tasks = [send_image(image_path) for image_path in image_paths]
    await asyncio.gather(*tasks)

asyncio.run(main())