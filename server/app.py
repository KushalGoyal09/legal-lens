from flask import Flask, request, jsonify
from flask_cors import CORS
from analyze import analyze_pdf_risk

app = Flask(__name__)
CORS(app)  

@app.route('/api/report', methods=['POST'])
def get_url_report():
    """API endpoint to generate a report from a URL."""
    request_data = request.get_json()
    
    if not request_data or 'url' not in request_data:
        return jsonify({'error': 'URL is required'}), 400
    
    url = request_data['url']
    
    report = analyze_pdf_risk(url)
    
    if 'error' in report:
        return jsonify(report), 400
    
    return jsonify(report), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)