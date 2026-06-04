# ═══════════════════════════════════════════════════════════════
#  AlzGlasses OCR Server
#  Receives a photo, reads the text in it, sends text back
#  Runs free on Render.com
# ═══════════════════════════════════════════════════════════════

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
import anthropic
import base64
import os

app = FastAPI()

# Allow requests from your Netlify app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your netlify URL later
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Anthropic client — reads text from images using Claude Vision
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Simple API key check so strangers can't use your server
API_KEY = os.environ.get("APP_API_KEY", "changeme")

@app.get("/")
def root():
    return {"status": "AlzGlasses OCR server is running"}

@app.post("/ocr")
async def ocr(
    image: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    # Check API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Read image bytes
    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image")

    # Convert to base64 for Claude Vision
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    # Detect image type
    content_type = image.content_type or "image/jpeg"

    # Ask Claude to read the text
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": content_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Please read all the text visible in this image, "
                            "in the natural reading order (left to right, top to bottom). "
                            "Return ONLY the text itself — no commentary, no formatting, "
                            "no explanation. If there is no text, return exactly: NO_TEXT"
                        )
                    }
                ],
            }
        ],
    )

    text = message.content[0].text.strip()

    # If Claude found no text
    if text == "NO_TEXT" or len(text) == 0:
        return {"text": "", "found": False}

    return {"text": text, "found": True}
