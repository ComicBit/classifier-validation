import os
import sys
import time
import json
import torch
import pandas as pd
from flask import Flask, request, render_template_string, redirect, url_for, Response, stream_with_context
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

app = Flask(__name__)
model = None
tokenizer = None
history = []  # list of tuples (question, answer)
current_model = None  # full path of the currently loaded model

# === Model Loading Functions ===
def load_model_from_path(model_path):
    global model, tokenizer, current_model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    current_model = model_path

def list_training_models():
    trainings_dir = "./trainings"
    models = []
    if os.path.exists(trainings_dir) and os.path.isdir(trainings_dir):
        for item in os.listdir(trainings_dir):
            item_path = os.path.join(trainings_dir, item)
            if os.path.isdir(item_path):
                models.append(item)
    return sorted(models)

# === Predict Batch (for CSV evaluation) ===
def predict_batch(texts, tokenizer, model, device, max_length=128):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1)
    return preds.cpu().numpy()

# === Routes for Main Interface ===
@app.route("/", methods=["GET", "POST"])
def index():
    global history, model
    result = None
    question = ""
    error_msg = ""
    if request.method == "POST":
        # Handle question submission from the Questions tab.
        if "submit_question" in request.form:
            question = request.form.get("question", "")
            if not model:
                error_msg = "No model loaded. Please load a model first."
            elif question:
                inputs = tokenizer(question, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**inputs).logits
                predicted_class = torch.argmax(logits, dim=1).item()
                result = "true" if predicted_class == 1 else "false"
                history.insert(0, (question, result))
    training_models = list_training_models()
    current_model_name = os.path.basename(current_model) if current_model else ""
    return render_template_string(MAIN_TEMPLATE, result=result, question=question, history=history,
                                  training_models=training_models, current_model=current_model_name,
                                  error_msg=error_msg)

@app.route("/load_model", methods=["POST"])
def load_model_route():
    model_name = request.form.get("model_name")
    if model_name:
        model_path = os.path.join("./trainings", model_name)
        try:
            load_model_from_path(model_path)
        except Exception as e:
            history.insert(0, (f"Error loading model {model_name}: {e}", ""))
    return redirect(url_for("index"))

@app.route("/shutdown", methods=["POST"])
def shutdown():
    shutdown_server()
    return "Server shutting down..."

def shutdown_server():
    func = request.environ.get("werkzeug.server.shutdown")
    if func is not None:
        func()
    os._exit(0)

# === JSON Evaluation Route ===
@app.route("/evaluate_json", methods=["POST"])
def evaluate_json():
    if 'json_file' not in request.files:
        return "No file provided", 400
    file = request.files['json_file']
    try:
        questions_data = json.loads(file.read().decode("utf-8"))
    except Exception as e:
        return f"Failed to parse JSON: {e}", 400

    if not model:
        return "No model loaded. Please load a model first.", 400

    return Response(stream_with_context(evaluate_json_generator(questions_data)),
                    mimetype='text/html')

def evaluate_json_generator(questions_data):
    try:
        total = len(questions_data)
    except Exception as e:
        yield f"<p>Error reading JSON data: {e}</p>"
        return

    correct_total = 0
    all_targets = []
    all_predictions = []
    current_model_name = os.path.basename(current_model) if current_model else "None"

    yield f"""<html>
<head>
  <title>JSON Evaluation</title>
  <style>
    html, body {{ margin: 0; padding: 0; height: 100%; font-family: Arial, sans-serif; }}
    .header {{
      position: sticky; top: 0;
      background: #fff; padding: 10px 20px;
      box-shadow: 0 2px 5px rgba(0,0,0,0.1);
      display: flex; justify-content: space-between; align-items: center;
      z-index: 1000;
    }}
    .header .model-info {{ font-weight: bold; }}
    .header button {{
      padding: 10px 20px; border: none; border-radius: 4px;
      background: #dc3545; color: #fff; cursor: pointer;
    }}
    .container {{
      height: calc(100% - 60px); display: flex; flex-direction: column;
    }}
    .log-container {{
      flex: 1; overflow-y: auto; padding: 20px;
      background: #f4f4f4; font-family: monospace; white-space: pre-wrap;
    }}
    .stats {{
      padding: 10px 20px; background: #fff; border-top: 1px solid #ddd; text-align: center;
    }}
    .stats a.back-button {{
      display: inline-block; margin-top: 10px; padding: 10px 20px;
      background: #007BFF; color: #fff; text-decoration: none; border-radius: 4px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <div class="model-info">Model: {current_model_name}</div>
    <form method="POST" action="/shutdown">
      <button type="submit">Quit</button>
    </form>
  </div>
  <div class="container">
    <div class="log-container" id="log">
      Total questions to evaluate: {total}\n\n
"""
    try:
        for idx, q in enumerate(questions_data, start=1):
            text = q.get("Question", "")
            target = 1 if q.get("Medical", False) else 0
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            with torch.no_grad():
                logits = model(**inputs).logits
            prediction = torch.argmax(logits, dim=1).item()
            all_predictions.append(prediction)
            all_targets.append(target)
            if prediction == target:
                correct_total += 1
            color = "green" if prediction == target else "red"
            pred_label = "Medical Advice" if prediction == 1 else "Non-Medical Advice"
            target_label = "Medical Advice" if target == 1 else "Non-Medical Advice"
            yield f'<span style="color: {color};">Q{idx}: {text}</span>\n'
            yield f'   Prediction: {pred_label} | Ground Truth: {target_label}\n'
            progress = (idx / total) * 100
            yield f"   Progress: {idx}/{total} ({progress:.2f}%)\n\n"
            yield '<script>document.getElementById("log").scrollTop = document.getElementById("log").scrollHeight;</script>\n'
    except Exception as e:
        yield f"<p>Error during evaluation: {e}</p>"
        yield "</div></div></body></html>"
        return

    overall_accuracy = correct_total / total if total > 0 else 0
    final_accuracy = accuracy_score(all_targets, all_predictions)
    final_f1 = f1_score(all_targets, all_predictions, average="weighted")
    final_precision = precision_score(all_targets, all_predictions, average="weighted")
    final_recall = recall_score(all_targets, all_predictions, average="weighted")

    yield """    </div>
    <div class="stats">
"""
    yield f"Overall Accuracy: {overall_accuracy*100:.2f}% ({correct_total}/{total})<br>\n"
    yield "Final Evaluation Metrics:<br>\n"
    yield f"   Accuracy:  {final_accuracy:.4f}<br>\n"
    yield f"   F1 Score:  {final_f1:.4f}<br>\n"
    yield f"   Precision: {final_precision:.4f}<br>\n"
    yield f"   Recall:    {final_recall:.4f}<br>\n"
    yield '<a class="back-button" href="/">Back</a>'
    yield """    </div>
  </div>
</body>
</html>
"""

# === CSV Evaluation Route ===
@app.route("/evaluate_csv", methods=["POST"])
def evaluate_csv():
    if 'csv_file' not in request.files:
        return "No file provided", 400
    file = request.files['csv_file']
    try:
        # read CSV using pandas; expects columns "text" and "target"
        df = pd.read_csv(file)
    except Exception as e:
        return f"Failed to parse CSV: {e}", 400

    if not model:
        return "No model loaded. Please load a model first.", 400

    return Response(stream_with_context(evaluate_csv_generator(df)),
                    mimetype='text/html')

def evaluate_csv_generator(df, batch_size=32):
    try:
        texts = df["text"].astype(str).tolist()
        targets = df["target"].tolist()
        total_samples = len(texts)
    except Exception as e:
        yield f"<p>Error processing CSV data: {e}</p>"
        return

    correct = 0
    all_preds = []
    all_targets = []
    current_model_name = os.path.basename(current_model) if current_model else "None"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yield f"""<html>
<head>
  <title>CSV Evaluation</title>
  <style>
    html, body {{ margin: 0; padding: 0; height: 100%; font-family: Arial, sans-serif; }}
    .header {{
      position: sticky; top: 0;
      background: #fff; padding: 10px 20px;
      box-shadow: 0 2px 5px rgba(0,0,0,0.1);
      display: flex; justify-content: space-between; align-items: center;
      z-index: 1000;
    }}
    .header .model-info {{ font-weight: bold; }}
    .header button {{
      padding: 10px 20px; border: none; border-radius: 4px;
      background: #dc3545; color: #fff; cursor: pointer;
    }}
    .container {{
      height: calc(100% - 60px); display: flex; flex-direction: column;
    }}
    .log-container {{
      flex: 1; overflow-y: auto; padding: 20px;
      background: #f4f4f4; font-family: monospace; white-space: pre-wrap;
    }}
    .stats {{
      padding: 10px 20px; background: #fff; border-top: 1px solid #ddd; text-align: center;
    }}
    .stats a.back-button {{
      display: inline-block; margin-top: 10px; padding: 10px 20px;
      background: #007BFF; color: #fff; text-decoration: none; border-radius: 4px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <div class="model-info">Model: {current_model_name}</div>
    <form method="POST" action="/shutdown">
      <button type="submit">Quit</button>
    </form>
  </div>
  <div class="container">
    <div class="log-container" id="log">
      Total samples to evaluate: {total_samples}\n\n
"""
    for i in range(0, total_samples, batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        preds = predict_batch(batch_texts, tokenizer, model, device)
        all_preds.extend(preds.tolist())
        all_targets.extend(batch_targets)
        batch_correct = sum(1 for p, t in zip(preds, batch_targets) if p == t)
        correct += batch_correct
        # yield details for each sample in this batch
        for j, (text, target, pred) in enumerate(zip(batch_texts, batch_targets, preds), start=i+1):
            color = "green" if pred == target else "red"
            pred_label = "Medical Advice" if pred == 1 else "Non-Medical Advice"
            target_label = "Medical Advice" if target == 1 else "Non-Medical Advice"
            yield f'<span style="color: {color};">Sample {j}: {text}</span>\n'
            yield f'   Prediction: {pred_label} | Ground Truth: {target_label}\n'
        running_accuracy = correct / (i + len(batch_texts))
        yield f"Running Accuracy: {running_accuracy:.4f} ({correct}/{i + len(batch_texts)})\n"
        progress = ((i + len(batch_texts)) / total_samples) * 100
        yield f"Progress: {i + len(batch_texts)}/{total_samples} ({progress:.2f}%)\n\n"
        yield '<script>document.getElementById("log").scrollTop = document.getElementById("log").scrollHeight;</script>\n'
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="weighted")
    precision = precision_score(all_targets, all_preds, average="weighted")
    recall = recall_score(all_targets, all_preds, average="weighted")

    yield """    </div>
    <div class="stats">
"""
    yield f"Final Accuracy: {accuracy*100:.2f}%<br>\n"
    yield f"F1 Score: {f1:.4f}<br>\n"
    yield f"Precision: {precision:.4f}<br>\n"
    yield f"Recall: {recall:.4f}<br>\n"
    yield '<a class="back-button" href="/">Back</a>'
    yield """    </div>
  </div>
</body>
</html>
"""

# === MAIN TEMPLATE with Three Tabs ===
MAIN_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Boolean Model Interface</title>
  <style>
    body { 
      font-family: Arial, sans-serif; 
      background-color: #f4f4f4; 
      margin: 0; padding: 0;
    }
    /* Toolbar styles */
    .toolbar-left {
      position: fixed; top: 20px; left: 20px;
      display: flex; align-items: center; gap: 10px; z-index: 1000;
    }
    .toolbar-right {
      position: fixed; top: 20px; right: 20px; z-index: 1000;
    }
    #quit-button {
      background: #dc3545; color: white; border: none;
      padding: 10px 20px; border-radius: 4px; cursor: pointer;
    }
    #quit-button:hover { background: #c82333; }
    #load-button {
      padding: 10px 20px; border: none;
      background: #28a745; color: white; border-radius: 4px; cursor: pointer;
    }
    #load-button:disabled {
      background: #6c757d; cursor: not-allowed;
    }
    /* Container */
    .container {
      max-width: 600px; margin: 140px auto 40px;
      padding: 20px; background: #fff;
      border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    input[type="text"], input[type="file"] {
      width: 100%; padding: 10px; margin: 10px 0;
      border: 1px solid #ccc; border-radius: 4px;
    }
    input[type="submit"] {
      padding: 10px 20px; border: none;
      background: #007BFF; color: white;
      border-radius: 4px; cursor: pointer;
    }
    input[type="submit"]:hover { background: #0056b3; }
    .result { margin-top: 20px; font-size: 1.2em;
      opacity: 0; transition: opacity 1s ease-in;
    }
    .result.show { opacity: 1; }
    .history { margin-top: 40px; }
    .history h3 { margin-bottom: 10px; }
    .history-item { padding: 10px; border-bottom: 1px solid #ddd; }
    .history-item:last-child { border-bottom: none; }
    .question { font-weight: bold; }
    .answer { margin-left: 10px; color: #007BFF; }
    .current-model { margin-top: 10px;
      font-size: 0.9em; color: #555;
    }
    .error-msg { margin-top: 10px;
      color: #dc3545; font-weight: bold;
    }
    /* Custom dropdown */
    .dropdown {
      position: relative; display: inline-block; min-width: 200px;
    }
    .dropdown-selected {
      padding: 8px 12px; border: 1px solid #ccc;
      border-radius: 4px; background-color: #fff;
      cursor: pointer; user-select: none;
    }
    .dropdown-options {
      display: none; position: absolute; background-color: #fff;
      border: 1px solid #ccc; border-radius: 4px;
      margin-top: 2px; z-index: 100; width: 100%;
      max-height: 200px; overflow-y: auto;
    }
    .dropdown-options .dropdown-option {
      padding: 8px 12px; cursor: pointer;
    }
    .dropdown-options .dropdown-option:hover {
      background-color: #f1f1f1;
    }
    .dropdown.show .dropdown-options {
      display: block;
    }
    /* Tab styles */
    .tab { display: none; }
    .tab.active { display: block; }
    /* Floating buttons for tabs */
    .floating-buttons {
      position: fixed; bottom: 20px; left: 50%;
      transform: translateX(-50%); display: flex;
      gap: 20px; z-index: 1000;
    }
    .floating-buttons button {
      padding: 10px 20px; border: none;
      background: #007BFF; color: #fff;
      border-radius: 4px; cursor: pointer;
    }
    .floating-buttons button:hover {
      background: #0056b3;
    }
  </style>
</head>
<body>
  <!-- Top toolbars -->
  <div class="toolbar-left">
    <form id="model-load-form" method="POST" action="/load_model" onsubmit="disableLoadButton()">
      <div class="dropdown" id="custom-dropdown">
        <div class="dropdown-selected" id="dropdown-selected">
          {{ current_model if current_model else "Select Model" }}
        </div>
        <div class="dropdown-options" id="dropdown-options">
          {% for m in training_models %}
            <div class="dropdown-option" data-value="{{ m }}">{{ m }}</div>
          {% endfor %}
        </div>
      </div>
      <input type="hidden" name="model_name" id="model-name-input" value="{{ current_model }}">
      <button id="load-button" type="submit">Load</button>
    </form>
  </div>
  <div class="toolbar-right">
    <form id="quit-form" method="POST" action="/shutdown">
      <button id="quit-button" type="submit">Quit</button>
    </form>
  </div>
  <!-- Main content container with three tabs -->
  <div class="container">
    <!-- Questions Tab -->
    <div id="questions-tab" class="tab active">
      <h2>Ask the Boolean Model</h2>
      <div class="current-model">Currently loaded model: <strong>{{ current_model if current_model else "None" }}</strong></div>
      <form id="question-form" method="POST">
        <input type="text" name="question" placeholder="Enter your question" value="{{ question }}" autofocus>
        <input type="submit" name="submit_question" value="Submit">
      </form>
      {% if error_msg %}
        <div class="error-msg">{{ error_msg }}</div>
      {% endif %}
      {% if result is not none %}
        <div id="result" class="result show">
          Response: <strong>{{ result }}</strong>
        </div>
      {% endif %}
      {% if history %}
        <div class="history">
          <h3>History</h3>
          {% for q, a in history %}
            <div class="history-item">
              <span class="question">{{ q }}</span> 
              {% if a %} - <span class="answer">{{ a }}</span>{% endif %}
            </div>
          {% endfor %}
        </div>
      {% endif %}
    </div>
    <!-- JSON Tab -->
    <div id="json-tab" class="tab">
      <h2>Test Model with JSON</h2>
      <p>Upload a JSON file with questions (each object should contain a "Question" field and a boolean "Medical" field):</p>
      <form id="json-form" method="POST" action="/evaluate_json" enctype="multipart/form-data">
        <input type="file" name="json_file" accept=".json" required>
        <input type="submit" value="Test JSON">
      </form>
      <p>Note: after submitting, you'll be redirected to a page showing real-time logs of the evaluation.</p>
    </div>
    <!-- CSV Tab -->
    <div id="csv-tab" class="tab">
      <h2>Test Model with CSV</h2>
      <p>Upload a CSV file with questions (expects columns "text" and "target"):</p>
      <form id="csv-form" method="POST" action="/evaluate_csv" enctype="multipart/form-data">
        <input type="file" name="csv_file" accept=".csv" required>
        <input type="submit" value="Test CSV">
      </form>
      <p>Note: after submitting, you'll be redirected to a page showing real-time logs of the evaluation.</p>
    </div>
  </div>
  <!-- Floating buttons to switch tabs -->
  <div class="floating-buttons">
    <button onclick="showTab('questions-tab')">Questions</button>
    <button onclick="showTab('csv-tab')">CSV</button>
    <button onclick="showTab('json-tab')">JSONiN</button>
  </div>
  <script>
    // Custom dropdown functionality
    const dropdown = document.getElementById("custom-dropdown");
    const dropdownSelected = document.getElementById("dropdown-selected");
    const dropdownOptions = document.getElementById("dropdown-options");
    const hiddenInput = document.getElementById("model-name-input");

    dropdownSelected.addEventListener("click", function() {
      dropdown.classList.toggle("show");
    });

    document.querySelectorAll(".dropdown-option").forEach(function(option) {
      option.addEventListener("click", function() {
        const value = this.getAttribute("data-value");
        dropdownSelected.textContent = value;
        hiddenInput.value = value;
        dropdown.classList.remove("show");
      });
    });

    window.addEventListener("click", function(e) {
      if (!dropdown.contains(e.target)) {
        dropdown.classList.remove("show");
      }
    });

    function disableLoadButton() {
      document.getElementById("load-button").disabled = true;
    }
    // Tab switching functionality
    function showTab(tabId) {
      document.querySelectorAll('.tab').forEach(function(tab) {
        tab.classList.remove('active');
      });
      document.getElementById(tabId).classList.add('active');
    }
    // Fade-in effect for result div
    document.addEventListener("DOMContentLoaded", function() {
      var resultDiv = document.getElementById("result");
      if (resultDiv) {
        resultDiv.classList.remove("show");
        setTimeout(function() {
          resultDiv.classList.add("show");
        }, 100);
      }
    });
  </script>
</body>
</html>
'''

if __name__ == '__main__':
    # No default model loaded on startup.
    app.run(host="0.0.0.0", port=3009)
