from flask import Flask, request, jsonify
import os
import time
from flask_cors import CORS
from assessment_compliance import load_pretrained_model, predict_document
from assessment_verification import verifyDocument
import requests
import tempfile

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://maroon-gerbil-893812.hostingersite.com",
            "http://localhost:3000" 
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})




model = None

def get_model():
    global model
    if model is None:
        model = load_pretrained_model()
    return model
    
from functools import wraps

def safe_api(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            app.logger.error(f"API Error: {str(e)}")
            return jsonify({"error": "Internal server error", "details": str(e)}), 500
    return wrapper


@app.route('/assess_compliance_risk', methods=['POST'])
@safe_api
def assess_compliance_risk():
    print("Received POST request to /assess_compliance_risk")
    print("Request data:", request.json)
    start_time = time.time()
    
    if not request.json or 'companyName' not in request.json:
        return jsonify({"error": "companyName is required"}), 400

    data = request.get_json()
    company_name = data.get('companyName')
    file_paths = data.get('documents', {})
    
    if not file_paths:
        return jsonify({"error": "No files found for the given companyName"}), 404

    file_types_to_process = ['BIR_permit', 'GIS', 'financial_statements']
    results = {
        doc_type: {"error": "Document not provided", "status": "missing"} 
        for doc_type in file_types_to_process
    }

    for file_type in file_types_to_process:
        file_path = file_paths.get(file_type)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        raw_path = os.path.join(script_dir,'..', 'FRONT_END')
        full_path = os.path.normpath(os.path.join(raw_path, file_path))
        print(full_path)

        if not full_path or not isinstance(full_path, str):
            continue
            
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
            if os.path.exists(full_path):
                try:
                    results[file_type] = predict_document(full_path, file_type, get_model(), company_name)
                except Exception as e:
                    results[file_type] = {
                        "error": str(e),
                        "status": "processing_error"
                    }
            else:
                results[file_type] = {
                    "error": f"File not found at {full_path}",
                    "status": "file_not_found"
                }

    def extract_score(result):
        if not isinstance(result, dict):
            return None
        score_str = result.get('final_score') or result.get('total_score')
        if not score_str:
            return None
        try:
            return float(str(score_str).strip('%'))
        except (ValueError, AttributeError):
            return None

    valid_scores = [score for score in map(extract_score, results.values()) if score is not None]
    
    if valid_scores:
        overall_score = sum(valid_scores) / len(valid_scores)
        risk_level = (
            "Low" if overall_score >= 70 
            else "Medium" if overall_score >= 40 
            else "High"
        )
    else:
        overall_score = 0
        risk_level = "High (No valid documents)"

    return jsonify({
        "company_name": company_name,
        "document_results": results,
        "overall_authenticity_score": f"{overall_score:.2f}%",
        "risk_level": risk_level,
        "processing_time_seconds": round(time.time() - start_time, 2)
    }), 200

@app.route('/verify', methods=['POST'])
@safe_api
def verify():
    print("Received POST request to /verification")
    print("Request data:", request.json)
    start_time = time.time()
    
    if not request.json or 'companyName' not in request.json:
        return jsonify({"error": "companyName is required"}), 400

    data = request.get_json()
    company_name = data.get('companyName')
    file_paths = data.get('documents', {})

    if not file_paths:
        return jsonify({"error": "No files found for the given companyName"}), 404

    file_types_to_process = ['GIS', 'BIR_permit', 'financial_statements']
    results = {
        doc_type: {"error": "Document not provided", "status": "missing"} 
        for doc_type in file_types_to_process
    }

    for file_type in file_types_to_process:
        file_url = file_paths.get(file_type)
        
        if not file_url or not isinstance(file_url, str):
            continue
            
        try:
            response = requests.get(file_url)
            response.raise_for_status()  
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_url)[1]) as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name
            
            try:
                results[file_type] = verifyDocument(tmp_file_path, file_type, get_model(), company_name)
            except Exception as e:
                results[file_type] = {
                    "error": str(e),
                    "status": "processing_error"
                }
            finally:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
        except requests.exceptions.RequestException as e:
            results[file_type] = {
                "error": f"Failed to download file from {file_url}: {str(e)}",
                "status": "download_error"
            }
        except Exception as e:
            results[file_type] = {
                "error": str(e),
                "status": "processing_error"
            }

    def extract_score(result):
        if not isinstance(result, dict):
            return None
        score_str = result.get('final_score')
        if not score_str:
            return None
        try:
            return float(str(score_str).strip('%'))
        except (ValueError, AttributeError):
            return None

    valid_scores = [score for score in map(extract_score, results.values()) if score is not None]
    
    if valid_scores:
        overall_score = sum(valid_scores) / len(valid_scores)
        risk_level = (
            "Valid" if overall_score >= 70 
            else "Medium Validatity" if overall_score >= 40 
            else "Invalid"
        )
    else:
        overall_score = 0
        risk_level = "Invalid (No valid documents)"

    return jsonify({
        "company_name": company_name,
        "document_results": results,
        "overall_authenticity_score": f"{overall_score:.2f}%",
        "risk_level": risk_level,
        "processing_time_seconds": round(time.time() - start_time, 2)
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)