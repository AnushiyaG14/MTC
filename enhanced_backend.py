from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from enhanced_database import EnhancedDocumentDatabase  # Fixed import
from document_parser import DocumentParser  # Fixed import
import threading
import sys
import json

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize services
db = EnhancedDocumentDatabase()  # Using original class name
parser = DocumentParser()  # Using fixed class name

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'message': 'Backend is running',
        'ml_enabled': True,
        'qwen_model_available': hasattr(parser, 'ml_trainer') and parser.ml_trainer.qwen_model is not None
    })

@app.route('/api/parse', methods=['POST'])
def parse_document():
    """API endpoint to parse uploaded document with ML enhancement"""
    try:
        print("Received parse request with ML enhancement")
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        print(f"Processing file: {file.filename}")
        
        # Read file content
        file_content = file.read()
        file_type = file.content_type
        filename = file.filename
        
        print(f"File type: {file_type}, Size: {len(file_content)} bytes")
        
        # Check if training mode is enabled
        enable_training = request.form.get('enable_training', 'true').lower() == 'true'
        
        # Process with enhanced ML parser
        result = parser.process_document(file_content, file_type, filename, enable_training)
        
        if 'error' in result:
            print(f"Processing error: {result['error']}")
            return jsonify({'success': False, 'error': result['error']}), 500
        
        print("Document processed successfully with ML enhancement")
        
        # Store in MongoDB with enhanced data
        doc_id = db.store_document(result)
        
        if doc_id:
            result['document_id'] = doc_id
            print(f"Stored in database with ID: {doc_id}")
            return jsonify({'success': True, 'data': result})
        else:
            return jsonify({
                'success': False, 
                'error': 'Failed to store in database'
            }), 500
            
    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({
            'success': False, 
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """API endpoint to manually trigger model training"""
    try:
        print("Received model training request")
        
        # Train the field classifier
        if hasattr(parser, 'ml_trainer'):
            success = parser.ml_trainer.train_field_classifier()
            
            if success:
                return jsonify({
                    'success': True, 
                    'message': 'Model training completed successfully'
                })
            else:
                return jsonify({
                    'success': False, 
                    'error': 'Model training failed - insufficient data or other error'
                }), 400
        else:
            return jsonify({
                'success': False, 
                'error': 'ML trainer not available'
            }), 400
            
    except Exception as e:
        print(f"Training error: {str(e)}")
        return jsonify({
            'success': False, 
            'error': f'Training error: {str(e)}'
        }), 500

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Get current model status and training data statistics"""
    try:
        if hasattr(parser, 'ml_trainer'):
            training_data = parser.ml_trainer.load_training_data()
            
            status = {
                'total_training_samples': len(training_data),
                'valid_samples': len([s for s in training_data if s.get('is_valid', False)]),
                'field_classifier_trained': parser.ml_trainer.field_classifier is not None,
                'vectorizer_trained': parser.ml_trainer.vectorizer is not None,
                'qwen_model_loaded': parser.ml_trainer.qwen_model is not None,
                'confidence_threshold': parser.ml_trainer.confidence_threshold
            }
        else:
            status = {
                'total_training_samples': 0,
                'valid_samples': 0,
                'field_classifier_trained': False,
                'vectorizer_trained': False,
                'qwen_model_loaded': False,
                'confidence_threshold': 0.8
            }
        
        return jsonify({'success': True, 'status': status})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get all documents"""
    try:
        documents = db.get_all_documents()
        for doc in documents:
            doc['_id'] = str(doc['_id'])
        return jsonify({'success': True, 'documents': documents})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/documents/<doc_id>/heat-records', methods=['GET'])
def get_heat_records(doc_id):
    """Get heat records for a specific document"""
    try:
        heat_records = db.get_heat_records_by_document(doc_id)
        for record in heat_records:
            record['_id'] = str(record['_id'])
            record['document_id'] = str(record['document_id'])
        return jsonify({'success': True, 'heat_records': heat_records})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_documents():
    """Search documents"""
    try:
        query = request.args.get('q', '')
        documents = db.search_documents(query)
        for doc in documents:
            doc['_id'] = str(doc['_id'])
        return jsonify({'success': True, 'documents': documents})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics/extraction-accuracy', methods=['GET'])
def get_extraction_accuracy():
    """Get extraction accuracy analytics"""
    try:
        # Simple analytics for now
        documents = db.get_all_documents()
        total_docs = len(documents)
        successful_docs = len([d for d in documents if d.get('validation_status') == 'PASSED'])
        
        analytics = {
            'total_documents': total_docs,
            'successful_extractions': successful_docs,
            'success_rate': (successful_docs / total_docs * 100) if total_docs > 0 else 0,
            'recent_documents_30_days': total_docs,  # Simplified
            'extraction_methods': [
                {
                    '_id': 'openai',
                    'total_documents': total_docs,
                    'successful_extractions': successful_docs,
                    'avg_confidence': 0.8
                }
            ]
        }
        
        return jsonify({'success': True, 'analytics': analytics})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export/training-data', methods=['GET'])
def export_training_data():
    """Export training data for external analysis"""
    try:
        if hasattr(parser, 'ml_trainer'):
            training_data = parser.ml_trainer.load_training_data()
            
            # Remove sensitive information if needed
            cleaned_data = []
            for sample in training_data:
                cleaned_sample = {
                    'timestamp': sample.get('timestamp', ''),
                    'is_valid': sample.get('is_valid', False),
                    'validation_errors': sample.get('validation_result', {}).get('errors', []),
                    'has_corrections': bool(sample.get('user_corrections', {}))
                }
                cleaned_data.append(cleaned_sample)
        else:
            cleaned_data = []
        
        return jsonify({
            'success': True, 
            'training_data': cleaned_data,
            'total_samples': len(cleaned_data)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def run_flask():
    """Run Flask in a separate function"""
    port = int(os.getenv('FLASK_PORT', 5000))
    print(f"Starting Enhanced Flask server with ML capabilities on port {port}")
    
    # Initialize ML models on startup (optional, can be lazy-loaded)
    print("Initializing ML models...")
    try:
        if hasattr(parser, 'ml_trainer'):
            parser.ml_trainer.initialize_qwen_model()
            print("ML models initialized successfully")
        else:
            print("ML trainer not available - using basic parsing only")
    except Exception as e:
        print(f"Warning: Could not initialize ML models: {str(e)}")
        print("Will fallback to OpenAI-only parsing")
    
    app.run(debug=False, port=port, host='127.0.0.1', use_reloader=False, threaded=True)

if __name__ == '__main__':
    try:
        run_flask()
    except KeyboardInterrupt:
        print("Server stopped by user")
        sys.exit(0)