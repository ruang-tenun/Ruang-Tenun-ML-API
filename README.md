# Ruang Tenun Machine Learning API

Ruang Tenun Machine Learning API is a tool for automatically detecting, analyzing, and classifying various types of woven fabrics using machine learning technology.

## Development

### Requirements

- Python 3.x
- Flask
- Other required dependencies (specified in the `requirements.txt` file)

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1. **Set the FLASK_APP environment variable:**

    ```bash
    export FLASK_APP=app.py  # On Windows use `set FLASK_APP=app.py`
    ```

2. **Run the Flask application:**

    ```bash
    flask run
    ```

The application will be available at `http://127.0.0.1:5000/`.

## Endpoint

### Tenun Detection

- **URL:** `http://127.0.0.1:5000/predict`
- **Method:** `POST`
- **Description:** This endpoint accepts an image of woven fabric and returns the classification result along with a confidence score.

#### Request
- **Content-Type:** `multipart/form-data` (for file uploads)
- **Data Params:**

    ```json
    {
        "image": <file image>
    }
    ```
The `image` parameter should be a file (JPEG, PNG, etc.) representing the woven fabric you want to classify.

#### Response
- **Content-Type:** `application/json`
- **Body:**

    ```json
    {
        "confidence_score": 0.996875524520874,
        "created_at": "2024-12-12T13:35:08.962219",
        "id": "e9b1c91d-6b48-452f-bd0e-62ceb7bb15eb",
        "result": "Endek Bali"
    }
    ```

## Example Usage

### Using `curl`

**Send an image for prediction:**

```bash
curl -X POST -F 'image=@path/to/your/image.jpg' http://127.0.0.1:5000/predict
