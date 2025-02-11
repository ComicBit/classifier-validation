# Classifier Validation Interface

This is a Flask-based web application that provides an interactive interface for a binary (True/False) classification model using the Hugging Face Transformers library. The application allows you to load a pre-trained model, ask questions to the model, and perform batch evaluations using CSV or JSON files. It also displays benchmark statistics for each evaluation allowing you to compare perfomances and track history of performed tests.

## Features

- **Model Loading:**  
  Load a pre-trained model from the `./trainings` directory via an AJAX-based interface so that the page does not reload when a model is selected.

- **Interactive Predictions:**  
  Enter a text query on the "Questions" tab to receive a binary classification ("True" or "False") response from the model. Your query history is displayed on the same page.

- **Batch Evaluation:**  
  Evaluate the model in batch mode using:
  - **CSV Evaluation:** Upload a CSV file with two columns, `text` and `target`.
  - **JSON Evaluation:** Upload a JSON file containing an array of objects with `text` and `target` keys.
  
- **Benchmark History:**  
  View performance metrics from your evaluations, including accuracy, F1 score, precision, and recall.

- **Server Shutdown:**  
  A "Quit" button is provided to gracefully shut down the server.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <https://github.com/ComicBit/classifier-validation>
   cd <classifier-validation>
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   run:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Application:**
   Run the Flask application by executing:
   ```bash
   python app.py
   ```
   The app will run on `http://0.0.0.0:3009`.

2. **Load a Model:**
   - Place your pre-trained model directories in the `./models` folder.
   - On the main page, click the "Select Model" button and choose the desired model. The selection is handled via AJAX, so the page remains intact.

3. **Ask a Question:**
   - Navigate to the **Questions** tab.
   - Enter your query into the provided input field and submit it.
   - The model will respond with "True" or "False", and your question/answer history will be updated.

4. **Evaluate the Model:**
   - **CSV Evaluation:**
     - Navigate to the **CSV** tab.
     - Upload a CSV file with the following format:
       ```csv
       text,target
       "Is this a valid statement?",1
       "Is this an invalid statement?",0
       ```
     - Click **Test CSV** to run the evaluation.
   
   - **JSON Evaluation:**
     - Navigate to the **JSON** tab.
     - Upload a JSON file with the following format:
       ```json
       [
         {"text": "Is this a valid statement?", "target": true},
         {"text": "Is this an invalid statement?", "target": false}
       ]
       ```
     - Click **Test JSON** to run the evaluation.

5. **View Benchmark History:**
   After each evaluation, performance metrics (accuracy, F1 score, precision, recall) are added to the Benchmark History section for review.

6. **Shutdown the Server:**
   Click the **Quit** button in the navigation bar to immediately shut down the server.

- **HTML & JavaScript:**
  - The HTML is rendered inline using Flask's `render_template_string` and leverages Tailwind CSS for styling. JavaScript handles tab switching, AJAX calls for model loading, auto-scrolling logs, and modal behavior for server shutdown.

## Customization

- **Model Domain:**  
  The application is designed for a generic binary classification. Modify the prediction logic if you need to support a different domain or additional classes.

- **User Interface:**  
  The UI uses Tailwind CSS for styling. Adjust the embedded styles or templates to match your branding.

- **Error Handling:**  
  Basic error handling is implemented throughout the application. Enhance it as necessary for your production environment.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- [Flask](https://flask.palletsprojects.com/) for the web framework.
- [Hugging Face Transformers](https://huggingface.co/transformers/) for model and tokenizer support.
- [Tailwind CSS](https://tailwindcss.com/) for the UI styling.

