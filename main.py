from fastapi import FastAPI, HTTPException, File, UploadFile
from deployment_inference import predict
import io

app = FastAPI()

@app.get("/")
def root():
    return {"Message":"Hello to FashionMNIST Fashion Clothing Predictor!"}

@app.post("/")
async def pred(img: UploadFile):
    image_content = await img.read()
    image_stream = io.BytesIO(image_content)

    return {"response": predict(image_stream)}