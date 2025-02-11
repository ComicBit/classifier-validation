import os
import json
import torch
import pandas as pd
from flask import Flask, request, render_template_string, Response, stream_with_context, abort, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

app = Flask(__name__)

# Global state
model = None
tokenizer = None
history = []              # List of (question, answer)
current_model = None      # Path of the currently loaded model
benchmark_results = []    # List of benchmark stats from evaluations


def load_model_from_path(model_path):
    global model, tokenizer, current_model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    current_model = model_path


def list_training_models():
    trainings_dir = "./models"
    if os.path.exists(trainings_dir) and os.path.isdir(trainings_dir):
        return sorted(
            [item for item in os.listdir(trainings_dir)
             if os.path.isdir(os.path.join(trainings_dir, item))]
        )
    return []


def predict_batch(texts, tokenizer, model, device, max_length=128):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1)
    return preds.cpu().numpy()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    question = ""
    error_msg = ""
    if request.method == "POST" and "submit_question" in request.form:
        question = request.form.get("question", "")
        if not model:
            error_msg = "No model loaded. Please load a model first."
        elif question:
            try:
                inputs = tokenizer(question, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**inputs).logits
                predicted_class = torch.argmax(logits, dim=1).item()
                result = "True" if predicted_class == 1 else "False"
                history.insert(0, (question, result))
            except Exception as e:
                error_msg = f"Error during prediction: {e}"

    training_models = list_training_models()
    current_model_name = os.path.basename(current_model) if current_model else ""
    return render_template_string(
        MAIN_TEMPLATE,
        result=result,
        question=question,
        history=history,
        training_models=training_models,
        current_model=current_model_name,
        error_msg=error_msg,
        benchmark_results=benchmark_results,
        navbar=navbar_html(),
    )


@app.route("/api/load_model", methods=["POST"])
def load_model_api():
    data = request.get_json()
    model_name = data.get("model_name")
    if model_name:
        model_path = os.path.join("./models", model_name)
        try:
            load_model_from_path(model_path)
            return jsonify({"status": "success", "model": model_name})
        except Exception as e:
            return jsonify({"status": "error", "message": f"Error loading model {model_name}: {e}"}), 500
    return jsonify({"status": "error", "message": "No model_name provided"}), 400


@app.route("/shutdown", methods=["POST"])
def shutdown():
    shutdown_server()
    return "Server shutting down..."


def shutdown_server():
    func = request.environ.get("werkzeug.server.shutdown")
    if func is not None:
        func()
    os._exit(0)


@app.route("/evaluate_json", methods=["POST"])
def evaluate_json():
    if "json_file" not in request.files:
        abort(400, description="No JSON file provided")
    file = request.files["json_file"]
    try:
        data = json.loads(file.read().decode("utf-8"))
    except Exception as e:
        abort(400, description=f"Failed to parse JSON: {e}")
    if not model:
        abort(400, description="No model loaded. Please load a model first.")
    return Response(
        stream_with_context(evaluate_json_generator(data)),
        mimetype="text/html"
    )


def evaluate_json_generator(data):
    try:
        total = len(data)
    except Exception as e:
        yield f"<p class='text-red-500'>Error reading JSON data: {e}</p>"
        return

    correct_total = 0
    all_targets = []
    all_predictions = []
    current_model_name = os.path.basename(current_model) if current_model else "None"

    yield f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>JSON Evaluation</title>
  <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100 min-h-screen">
  {navbar_html()}
  <main class="max-w-4xl mx-auto p-4 space-y-8">
    <div class="bg-white rounded-lg shadow-xl p-6">
      <div class="mb-4">
        <a href="/" class="inline-block bg-blue-500 hover:shadow-lg active:shadow-inner transition-shadow duration-500 text-white rounded-full py-2 px-4">Back</a>
      </div>
      <header>
        <h1 class="text-3xl font-bold text-gray-800">JSON Evaluation</h1>
        <p class="text-gray-600">Evaluating {total} entries.</p>
      </header>
      <div id="log" class="mt-4 bg-white rounded-lg shadow p-4 font-mono overflow-y-auto h-80">
        <p>Total entries to evaluate: {total}</p>
"""
    for idx, item in enumerate(data, start=1):
        text = item.get("text", "")
        raw_target = item.get("target", 0)
        # Convert target to integer: treat "true"/"1" as 1, else 0.
        target = 1 if str(raw_target).lower() in ["true", "1"] else 0
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            with torch.no_grad():
                logits = model(**inputs).logits
            prediction = torch.argmax(logits, dim=1).item()
        except Exception as e:
            yield f"<p class='text-red-500'>Error evaluating entry {idx}: {e}</p>"
            continue

        all_predictions.append(prediction)
        all_targets.append(target)
        if prediction == target:
            correct_total += 1

        color = "text-green-600" if prediction == target else "text-red-600"
        pred_label = "True" if prediction == 1 else "False"
        target_label = "True" if target == 1 else "False"
        yield f'<p class="{color}">Entry {idx}: {text}</p>'
        yield f'<p class="ml-4 text-sm">Prediction: {pred_label} | Ground Truth: {target_label}</p>'
        yield '<script>document.getElementById("log").scrollTop = document.getElementById("log").scrollHeight;</script>'

    overall_accuracy = correct_total / total if total > 0 else 0
    final_accuracy = accuracy_score(all_targets, all_predictions) if all_targets else 0
    final_f1 = f1_score(all_targets, all_predictions, average="weighted") if all_targets else 0
    final_precision = precision_score(all_targets, all_predictions, average="weighted") if all_targets else 0
    final_recall = recall_score(all_targets, all_predictions, average="weighted") if all_targets else 0

    benchmark_results.append({
        "mode": "JSON",
        "model": current_model_name,
        "overall_acc": overall_accuracy,
        "correct": correct_total,
        "total": total,
        "acc": final_accuracy,
        "f1": final_f1,
        "prec": final_precision,
        "rec": final_recall
    })

    yield f"""
      </div>
      <section class="mt-8">
        <div class="px-4">
          <h3 class="text-base font-semibold text-gray-900">Evaluation Report</h3>
          <p class="mt-1 text-sm text-gray-500">Performance benchmark for JSON evaluation on model <strong>{current_model_name}</strong>.</p>
        </div>
        <div class="mt-6 border-t border-gray-100">
          <dl class="divide-y divide-gray-100">
            <div class="px-4 py-6 grid grid-cols-3 gap-4">
              <dt class="text-sm font-medium text-gray-900">Overall Accuracy</dt>
              <dd class="text-sm text-gray-700 col-span-2">{overall_accuracy*100:.2f}% ({correct_total}/{total})</dd>
            </div>
            <div class="px-4 py-6 grid grid-cols-3 gap-4">
              <dt class="text-sm font-medium text-gray-900">Accuracy</dt>
              <dd class="text-sm text-gray-700 col-span-2">{final_accuracy:.4f}</dd>
            </div>
            <div class="px-4 py-6 grid grid-cols-3 gap-4">
              <dt class="text-sm font-medium text-gray-900">F1 Score</dt>
              <dd class="text-sm text-gray-700 col-span-2">{final_f1:.4f}</dd>
            </div>
            <div class="px-4 py-6 grid grid-cols-3 gap-4">
              <dt class="text-sm font-medium text-gray-900">Precision</dt>
              <dd class="text-sm text-gray-700 col-span-2">{final_precision:.4f}</dd>
            </div>
            <div class="px-4 py-6 grid grid-cols-3 gap-4">
              <dt class="text-sm font-medium text-gray-900">Recall</dt>
              <dd class="text-sm text-gray-700 col-span-2">{final_recall:.4f}</dd>
            </div>
          </dl>
        </div>
      </section>
    </div>
  </main>
</body>
</html>
"""


@app.route("/evaluate_csv", methods=["POST"])
def evaluate_csv():
    if "csv_file" not in request.files:
        abort(400, description="No CSV file provided")
    file = request.files["csv_file"]
    try:
        df = pd.read_csv(file)
    except Exception as e:
        abort(400, description=f"Failed to parse CSV: {e}")
    if not model:
        abort(400, description="No model loaded. Please load a model first.")
    return Response(
        stream_with_context(evaluate_csv_generator(df)),
        mimetype="text/html"
    )


def evaluate_csv_generator(df, batch_size=32):
    try:
        texts = df["text"].astype(str).tolist()
        targets = df["target"].tolist()
        total_samples = len(texts)
    except Exception as e:
        yield f"<p class='text-red-500'>Error processing CSV data: {e}</p>"
        return

    correct = 0
    all_preds = []
    all_targets = []
    current_model_name = os.path.basename(current_model) if current_model else "None"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yield f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CSV Evaluation</title>
  <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100 min-h-screen">
  {navbar_html()}
  <main class="max-w-4xl mx-auto p-4 space-y-8">
    <div class="bg-white rounded-lg shadow-xl p-6">
      <div class="mb-4">
        <a href="/" class="inline-block bg-blue-500 hover:shadow-lg active:shadow-inner transition-shadow duration-200 text-white rounded-full py-2 px-4">Back</a>
      </div>
      <header>
        <h1 class="text-3xl font-bold text-gray-800">CSV Evaluation</h1>
        <p class="text-gray-600">Evaluating {total_samples} samples.</p>
      </header>
      <div id="log" class="mt-4 bg-white rounded-lg shadow p-4 font-mono overflow-y-auto h-80">
        <p>Total samples to evaluate: {total_samples}</p>
"""
    for i in range(0, total_samples, batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        try:
            preds = predict_batch(batch_texts, tokenizer, model, device)
        except Exception as e:
            yield f"<p class='text-red-500'>Error processing batch starting at sample {i+1}: {e}</p>"
            continue

        all_preds.extend(preds.tolist())
        all_targets.extend(batch_targets)
        batch_correct = sum(1 for p, t in zip(preds, batch_targets) if p == t)
        correct += batch_correct

        for j, (text, target, pred) in enumerate(zip(batch_texts, batch_targets, preds), start=i+1):
            color = "text-green-600" if pred == target else "text-red-600"
            pred_label = "True" if pred == 1 else "False"
            target_label = "True" if target == 1 else "False"
            yield f'<p class="{color}">Sample {j}: {text}</p>'
            yield f'<p class="ml-4 text-sm">Prediction: {pred_label} | Ground Truth: {target_label}</p>'
        yield '<script>document.getElementById("log").scrollTop = document.getElementById("log").scrollHeight;</script>'

    accuracy_val = accuracy_score(all_targets, all_preds) if all_targets else 0
    f1_val = f1_score(all_targets, all_preds, average="weighted") if all_targets else 0
    precision_val = precision_score(all_targets, all_preds, average="weighted") if all_targets else 0
    recall_val = recall_score(all_targets, all_preds, average="weighted") if all_targets else 0

    benchmark_results.append({
        "mode": "CSV",
        "model": current_model_name,
        "overall_acc": accuracy_val,
        "correct": correct,
        "total": total_samples,
        "acc": accuracy_val,
        "f1": f1_val,
        "prec": precision_val,
        "rec": recall_val
    })

    yield f"""
      </div>
      <section class="mt-8">
        <div class="px-4">
          <h3 class="text-base font-semibold text-gray-900">Evaluation Report</h3>
          <p class="mt-1 text-sm text-gray-500">Performance benchmark for CSV evaluation on model <strong>{current_model_name}</strong>.</p>
        </div>
        <div class="mt-6 border-t border-gray-100">
          <dl class="divide-y divide-gray-100">
            <div class="px-4 py-6 grid grid-cols-3 gap-4">
              <dt class="text-sm font-medium text-gray-900">Final Accuracy</dt>
              <dd class="text-sm text-gray-700 col-span-2">{accuracy_val*100:.2f}%</dd>
            </div>
            <div class="px-4 py-6 grid grid-cols-3 gap-4">
              <dt class="text-sm font-medium text-gray-900">F1 Score</dt>
              <dd class="text-sm text-gray-700 col-span-2">{f1_val:.4f}</dd>
            </div>
            <div class="px-4 py-6 grid grid-cols-3 gap-4">
              <dt class="text-sm font-medium text-gray-900">Precision</dt>
              <dd class="text-sm text-gray-700 col-span-2">{precision_val:.4f}</dd>
            </div>
            <div class="px-4 py-6 grid grid-cols-3 gap-4">
              <dt class="text-sm font-medium text-gray-900">Recall</dt>
              <dd class="text-sm text-gray-700 col-span-2">{recall_val:.4f}</dd>
            </div>
          </dl>
        </div>
      </section>
    </div>
  </main>
</body>
</html>
"""


def navbar_html():
    return """
    <nav class="bg-white shadow">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex items-center">
            <h1 class="text-2xl font-bold text-gray-800">Model Interface</h1>
          </div>
          <div class="flex items-center space-x-4">
            <button id="nav-quit-btn" class="px-4 py-2 bg-red-500 hover:shadow-lg active:shadow-inner transition-shadow duration-200 text-white rounded-full">Quit</button>
          </div>
        </div>
      </div>
    </nav>
    """


MAIN_TEMPLATE = r'''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Model Interface</title>
  <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
  <style>
    html { transition: background-color 0.3s, color 0.3s; }
    .btn { transition: box-shadow 200ms, transform 100ms; }
    .btn:hover { box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .btn:active { transform: scale(0.98); box-shadow: 0 2px 3px rgba(0,0,0,0.1); }
  </style>
</head>
<body class="bg-gray-100 min-h-screen">
  {{ navbar|safe }}
  <main class="max-w-4xl mx-auto p-4 space-y-8">
    <section>
      <h2 class="text-2xl font-bold text-gray-800 mb-4">Load a Model</h2>
      <div class="relative flex justify-center">
        <button id="model-menu-button" type="button" class="inline-flex items-center gap-x-1 text-sm font-semibold text-gray-900">
          <span id="current-model">{{ current_model if current_model else "Select Model" }}</span>
          <svg class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path fill-rule="evenodd" d="M5.22 8.22a.75.75 0 011.06 0L10 11.94l3.72-3.72a.75.75 0 111.06 1.06l-4.25 4.25a.75.75 0 01-1.06 0L5.22 9.28a.75.75 0 010-1.06z" clip-rule="evenodd" />
          </svg>
        </button>
        <div id="model-menu" class="absolute left-1/2 -translate-x-1/2 top-full mt-2 z-10 origin-top hidden transform">
          <div class="flex-auto overflow-hidden rounded-lg bg-white shadow-lg ring-1 ring-gray-900/5">
            <div class="p-4">
              {% for m in training_models %}
              <button onclick="loadModel('{{ m }}')" class="group relative flex gap-x-6 rounded-lg p-4 hover:bg-gray-50 btn w-full text-left">
                <div class="flex h-11 w-11 flex-none items-center justify-center rounded-lg bg-gray-50">
                  <svg class="h-6 w-6 text-gray-600 group-hover:text-gray-800" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M10.5 6a7.5 7.5 0 107.5 7.5h-7.5V6z" />
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M13.5 10.5H21A7.5 7.5 0 0013.5 3v7.5z" />
                  </svg>
                </div>
                <div>
                  <span class="font-semibold text-gray-900">{{ m }}</span>
                  <p class="mt-1 text-gray-600 text-sm">Load model</p>
                </div>
              </button>
              {% endfor %}
            </div>
          </div>
        </div>
      </div>
    </section>
    
    <section class="bg-white rounded-lg shadow-xl">
      <div class="border-b border-gray-200">
        <ul class="flex">
          <li class="w-1/3">
            <button id="questions-tab-btn" onclick="switchTab('questions-tab')" class="tab-btn w-full py-4 text-lg font-semibold text-blue-500 border-b-2 border-blue-500">Questions</button>
          </li>
          <li class="w-1/3">
            <button id="csv-tab-btn" onclick="switchTab('csv-tab')" class="tab-btn w-full py-4 text-lg font-semibold text-gray-600">CSV</button>
          </li>
          <li class="w-1/3">
            <button id="json-tab-btn" onclick="switchTab('json-tab')" class="tab-btn w-full py-4 text-lg font-semibold text-gray-600">JSON</button>
          </li>
        </ul>
      </div>
      <div class="p-6">
        <div id="questions-tab" class="tab-content">
          <h2 class="text-2xl font-bold mb-4 text-gray-800">Ask the Model</h2>
          <p class="mb-4 text-gray-600">Enter a question below. The model will respond with <strong>True</strong> or <strong>False</strong>.</p>
          <form id="question-form" method="POST" class="space-y-4">
            <input type="text" name="question" placeholder="Enter your question" value="{{ question }}" autofocus class="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400">
            <input type="submit" name="submit_question" value="Submit" class="w-full py-3 bg-blue-500 hover:shadow-lg active:shadow-inner btn text-white rounded-full {{ 'opacity-50 cursor-not-allowed' if not current_model else '' }}" {% if not current_model %} disabled {% endif %}>
          </form>
          {% if error_msg %}
            <div class="mt-4 text-red-500 font-semibold">{{ error_msg }}</div>
          {% endif %}
          {% if result is not none %}
            <div id="result" class="mt-4 p-4 bg-gray-100 border-l-4 border-blue-500 text-xl font-semibold">
              Response: <span class="text-blue-700">{{ result }}</span>
            </div>
          {% endif %}
          {% if history %}
            <div class="mt-6">
              <h3 class="text-xl font-bold mb-2 text-gray-800">History</h3>
              <div class="space-y-2">
                {% for q, a in history %}
                  <div class="p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <span class="font-semibold">{{ q }}</span>{% if a %} - <span class="text-blue-500">{{ a }}</span>{% endif %}
                  </div>
                {% endfor %}
              </div>
            </div>
          {% endif %}
        </div>
        
        <div id="csv-tab" class="tab-content hidden">
          <h2 class="text-2xl font-bold mb-4 text-gray-800">Test Model with CSV</h2>
          <p class="mb-4 text-gray-600">Upload a CSV file to evaluate the model in batch mode. <strong>Expected format:</strong></p>
          <pre class="bg-gray-100 text-gray-900 p-3 rounded-lg shadow mb-4">
text,target
"Is this a valid statement?",1
"Is this an invalid statement?",0
          </pre>
          <form id="csv-form" method="POST" action="/evaluate_csv" enctype="multipart/form-data" class="space-y-4" onsubmit="return checkFileSelected('csv_file')">
            <div>
              <label for="csv_file" class="block text-sm font-medium text-gray-900">Upload CSV File</label>
              <div class="mt-2 flex justify-center rounded-lg border border-dashed border-gray-900/25 px-6 py-10">
                <div class="text-center">
                  <svg class="mx-auto h-12 w-12 text-gray-300" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                    <path fill-rule="evenodd" d="M1.5 6a2.25 2.25 0 012.25-2.25h16.5A2.25 2.25 0 0122.5 6v12a2.25 2.25 0 01-2.25 2.25H3.75A2.25 2.25 0 011.5 18V6zM3 16.06V18c0 .414.336.75.75.75h16.5a.75.75 0 00.75-.75v-1.94l-2.69-2.69a1.5 1.5 0 00-2.12 0l-.88.88.97.97a.75.75 0 11-1.06 1.06l-5.16-5.16a1.5 1.5 0 00-2.12 0L3 16.06zM13.125 8.25a1.125 1.125 0 112.25 0 1.125 1.125 0 01-2.25 0z" clip-rule="evenodd" />
                  </svg>
                  <div class="mt-4 flex text-sm text-gray-600">
                    <label for="csv_file" class="relative cursor-pointer rounded-md bg-white font-semibold text-blue-600 focus:ring-2 focus:ring-blue-600 focus:ring-offset-2 hover:shadow-lg btn">
                      <span>Upload a file</span>
                      <input id="csv_file" name="csv_file" type="file" class="sr-only" accept=".csv" required>
                    </label>
                    <p class="pl-1">or drag and drop</p>
                  </div>
                  <p class="text-xs text-gray-600">CSV file up to 10MB</p>
                </div>
              </div>
            </div>
            <input type="submit" value="Test CSV" class="w-full mt-4 py-3 bg-blue-500 hover:shadow-lg active:shadow-inner btn text-white rounded-full {{ 'opacity-50 cursor-not-allowed' if not current_model else '' }}" {% if not current_model %} disabled {% endif %}>
          </form>
        </div>
        
        <div id="json-tab" class="tab-content hidden">
          <h2 class="text-2xl font-bold mb-4 text-gray-800">Test Model with JSON</h2>
          <p class="mb-4 text-gray-600">Upload a JSON file to evaluate the model in batch mode. <strong>Expected format:</strong></p>
          <pre class="bg-gray-100 text-gray-900 p-3 rounded-lg shadow mb-4">
[
  {"text": "Is this a valid statement?", "target": true},
  {"text": "Is this an invalid statement?", "target": false}
]
          </pre>
          <form id="json-form" method="POST" action="/evaluate_json" enctype="multipart/form-data" class="space-y-4" onsubmit="return checkFileSelected('json_file')">
            <div>
              <label for="json_file" class="block text-sm font-medium text-gray-900">Upload JSON File</label>
              <div class="mt-2 flex justify-center rounded-lg border border-dashed border-gray-900/25 px-6 py-10">
                <div class="text-center">
                  <svg class="mx-auto h-12 w-12 text-gray-300" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                    <path fill-rule="evenodd" d="M1.5 6a2.25 2.25 0 012.25-2.25h16.5A2.25 2.25 0 0122.5 6v12a2.25 2.25 0 01-2.25 2.25H3.75A2.25 2.25 0 011.5 18V6zM3 16.06V18c0 .414.336.75.75.75h16.5a.75.75 0 00.75-.75v-1.94l-2.69-2.69a1.5 1.5 0 00-2.12 0l-.88.88.97.97a.75.75 0 11-1.06 1.06l-5.16-5.16a1.5 1.5 0 00-2.12 0L3 16.06zM13.125 8.25a1.125 1.125 0 112.25 0 1.125 1.125 0 01-2.25 0z" clip-rule="evenodd" />
                  </svg>
                  <div class="mt-4 flex text-sm text-gray-600">
                    <label for="json_file" class="relative cursor-pointer rounded-md bg-white font-semibold text-blue-600 focus:ring-2 focus:ring-blue-600 focus:ring-offset-2 hover:shadow-lg btn">
                      <span>Upload a file</span>
                      <input id="json_file" name="json_file" type="file" class="sr-only" accept=".json" required>
                    </label>
                    <p class="pl-1">or drag and drop</p>
                  </div>
                  <p class="text-xs text-gray-600">JSON file up to 10MB</p>
                </div>
              </div>
            </div>
            <input type="submit" value="Test JSON" class="w-full mt-4 py-3 bg-blue-500 hover:shadow-lg active:shadow-inner btn text-white rounded-full {{ 'opacity-50 cursor-not-allowed' if not current_model else '' }}" {% if not current_model %} disabled {% endif %}>
          </form>
        </div>
      </div>
    </section>
    
    {% if benchmark_results %}
    <section>
      <h2 class="text-2xl font-bold text-gray-800 mb-4">Benchmark History</h2>
      <div class="space-y-8">
        {% for bench in benchmark_results %}
        <div class="bg-white p-4 rounded-lg shadow">
          <div class="px-4">
            <h3 class="text-base font-semibold text-gray-900">{{ bench.mode }} Evaluation</h3>
            <p class="mt-1 text-sm text-gray-500">Tested on model: <strong>{{ bench.model }}</strong></p>
          </div>
          <div class="mt-6 border-t border-gray-100">
            <dl class="divide-y divide-gray-100">
              <div class="px-4 py-6 grid grid-cols-3 gap-4">
                <dt class="text-sm font-medium text-gray-900">Overall Accuracy</dt>
                <dd class="text-sm text-gray-700 col-span-2">{{ (bench.overall_acc * 100)|round(2) }}% ({{ bench.correct }}/{{ bench.total }})</dd>
              </div>
              <div class="px-4 py-6 grid grid-cols-3 gap-4">
                <dt class="text-sm font-medium text-gray-900">Accuracy</dt>
                <dd class="text-sm text-gray-700 col-span-2">{{ bench.acc|round(4) }}</dd>
              </div>
              <div class="px-4 py-6 grid grid-cols-3 gap-4">
                <dt class="text-sm font-medium text-gray-900">F1 Score</dt>
                <dd class="text-sm text-gray-700 col-span-2">{{ bench.f1|round(4) }}</dd>
              </div>
              <div class="px-4 py-6 grid grid-cols-3 gap-4">
                <dt class="text-sm font-medium text-gray-900">Precision</dt>
                <dd class="text-sm text-gray-700 col-span-2">{{ bench.prec|round(4) }}</dd>
              </div>
              <div class="px-4 py-6 grid grid-cols-3 gap-4">
                <dt class="text-sm font-medium text-gray-900">Recall</dt>
                <dd class="text-sm text-gray-700 col-span-2">{{ bench.rec|round(4) }}</dd>
              </div>
            </dl>
          </div>
        </div>
        {% endfor %}
      </div>
    </section>
    {% endif %}
  </main>
  
  <div id="quit-modal" class="fixed inset-0 z-10 hidden" aria-labelledby="modal-title" role="dialog" aria-modal="true">
    <div class="fixed inset-0 bg-gray-500/75 transition-opacity" aria-hidden="true"></div>
    <div class="fixed inset-0 z-10 w-screen overflow-y-auto">
      <div class="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
        <div class="relative transform overflow-hidden rounded-lg bg-white text-left shadow-xl transition-all sm:my-8 sm:w-full sm:max-w-lg">
          <div class="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
            <div class="sm:flex sm:items-start">
              <div class="mx-auto flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-red-100 sm:mx-0 sm:h-10 sm:w-10">
                <svg class="h-6 w-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
                </svg>
              </div>
              <div class="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                <h3 class="text-base font-semibold text-gray-900" id="modal-title">Confirm Quit</h3>
                <div class="mt-2">
                  <p class="text-sm text-gray-500">Are you sure you want to quit? This will immediately shut down the server.</p>
                </div>
              </div>
            </div>
          </div>
          <div class="bg-gray-50 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6">
            <button id="confirm-quit-btn" type="button" class="inline-flex w-full justify-center rounded-md bg-red-600 px-3 py-2 text-sm font-semibold text-white shadow hover:shadow-lg active:shadow-inner sm:ml-3 sm:w-auto">Quit</button>
            <button id="cancel-quit-btn" type="button" class="mt-3 inline-flex w-full justify-center rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow ring-1 ring-gray-300 hover:shadow-lg sm:mt-0 sm:w-auto">Cancel</button>
          </div>
        </div>
      </div>
    </div>
  </div>
  
  <script>
    // Model menu toggle
    const modelMenuButton = document.getElementById('model-menu-button');
    const modelMenu = document.getElementById('model-menu');
    modelMenuButton.addEventListener('click', function(event) {
      event.stopPropagation();
      modelMenu.classList.toggle('hidden');
    });
    document.addEventListener('click', function(event) {
      if (!modelMenu.contains(event.target)) {
        modelMenu.classList.add('hidden');
      }
    });
    
    // Tab switching with persistence
    function switchTab(tabId) {
      document.querySelectorAll(".tab-content").forEach(tab => tab.classList.add("hidden"));
      document.getElementById(tabId).classList.remove("hidden");
      document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.classList.remove("text-blue-500", "border-b-2", "border-blue-500");
        btn.classList.add("text-gray-600");
      });
      document.getElementById(tabId + "-btn").classList.remove("text-gray-600");
      document.getElementById(tabId + "-btn").classList.add("text-blue-500", "border-b-2", "border-blue-500");
      localStorage.setItem("currentTab", tabId);
    }
    document.addEventListener("DOMContentLoaded", function() {
      const savedTab = localStorage.getItem("currentTab") || "questions-tab";
      switchTab(savedTab);
    });
    
    function checkFileSelected(inputId) {
      const input = document.getElementById(inputId);
      if (!input.value) {
        alert("Please select a file before submitting.");
        return false;
      }
      return true;
    }
    
    // AJAX model loader
    function loadModel(modelName) {
      fetch("/api/load_model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelName })
      })
      .then(response => response.json())
      .then(data => {
        if (data.status === "success") {
          document.getElementById("current-model").innerText = data.model;
          document.querySelectorAll("input[type='submit']").forEach(btn => {
            btn.disabled = false;
            btn.classList.remove("opacity-50", "cursor-not-allowed");
          });
        } else {
          alert("Error loading model: " + data.message);
        }
      })
      .catch(error => {
        alert("Error loading model: " + error);
      });
    }
    
    // Quit modal behavior
    const navQuitBtn = document.getElementById("nav-quit-btn");
    const quitModal = document.getElementById("quit-modal");
    const cancelQuitBtn = document.getElementById("cancel-quit-btn");
    const confirmQuitBtn = document.getElementById("confirm-quit-btn");
    navQuitBtn.addEventListener("click", function(event) {
      event.preventDefault();
      quitModal.classList.remove("hidden");
    });
    cancelQuitBtn.addEventListener("click", function() {
      quitModal.classList.add("hidden");
    });
    confirmQuitBtn.addEventListener("click", function() {
      fetch("/shutdown", { method: "POST" }).then(() => {
        window.location.reload();
      });
    });
  </script>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3009)
