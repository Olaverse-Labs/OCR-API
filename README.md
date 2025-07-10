# Image to Text OCR API

A FastAPI-based OCR service that extracts text and structure from images, supporting advanced features like preprocessing, language selection, confidence scores, and more.

## Features
- Extracts text from uploaded images or image URLs
- Returns detected languages with confidence
- Returns bounding boxes for blocks, paragraphs, lines, and words
- Includes confidence scores for each word
- Supports image preprocessing: grayscale, thresholding, denoising
- Allows language selection for OCR
- Can automatically deskew (rotate) images

## Installation
1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) is installed and in your system PATH.

## Running the API
```bash
uvicorn main:app --reload
```
Or, if `uvicorn` is not in your PATH:
```bash
python -m uvicorn main:app --reload
```

## API Usage
Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive documentation.

### Endpoint: `/extract-text/` (POST)
Accepts either an image file upload or an image URL, with optional processing options.

#### Form Fields
- `file`: (optional) Image file to upload
- `image_url`: (optional) URL to an image
- `preprocess`: (optional) Comma-separated preprocessing steps: `grayscale`, `threshold`, `denoise`
- `ocr_lang`: (optional) Tesseract language(s), e.g. `eng`, `fra`, `eng+fra` (default: `eng`)
- `deskew`: (optional) `true` or `false` (default: `false`), deskew image before OCR

#### Example Request (using `curl`)
```bash
curl -X POST "http://localhost:8000/extract-text/" \
  -F "file=@your_image.png" \
  -F "preprocess=grayscale,threshold" \
  -F "ocr_lang=eng+fra" \
  -F "deskew=true"
```

#### Example Response
```json
{
  "status": true,
  "text": "...extracted text...",
  "boxCoordinates": [0.1, 0.2, 0.3, 0.4],
  "blocks": [
    {
      "boxCoordinates": [ ... ],
      "paragraphs": [
        {
          "boxCoordinates": [ ... ],
          "lines": [
            {
              "boxCoordinates": [ ... ],
              "text": "...",
              "words": [
                { "text": "...", "boxCoordinates": [ ... ], "confidence": 95.0 }
              ]
            }
          ]
        }
      ]
    }
  ],
  "detectedLanguages": [
    { "languageCode": "en", "confidence": 0.98 }
  ],
  "executionTimeMS": 1234
}
```

## Docker
A `Dockerfile` is provided for containerized deployment. Build and run with:
```bash
docker build -t image-to-text-api .
docker run -p 8000:8000 image-to-text-api
```

## Notes
- For best results, ensure Tesseract is installed and available in your system PATH.
- You can combine preprocessing options for improved OCR accuracy.
- The API supports both file uploads and image URLs.

## License
MIT 