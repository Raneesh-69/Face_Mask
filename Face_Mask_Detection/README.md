# Face Mask Detector (Static)

This project is a static frontend app for easy deployment on Netlify or Vercel.

## What It Does

- Upload an image
- Runs browser-side AI model inference
- Shows Mask or No Mask prediction with confidence

## Tech Stack

- HTML
- CSS
- JavaScript
- TensorFlow.js
- MobileNet (browser model)

## Project Structure

- index.html
- static/style.css
- static/app.js
- netlify.toml
- vercel.json

## Deploy

### Netlify

- Import the repository
- Build command: leave empty
- Publish directory: .

### Vercel

- Import the repository
- Framework preset: Other
- No build command required

## Notes

- This is fully static and does not require Python server deployment.
- Prediction behavior may differ from the original Flask plus TensorFlow backend model.
