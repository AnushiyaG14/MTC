from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class EnhancedDocumentDatabase:
    def __init__(self):
        self.client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
        self.db = self.client['document_parsing']
        self.documents_collection = self.db['documents']
        self.heat_records_collection = self.db['heat_records']
    
    def store_document(self, extracted_data):
        """Store document and heat records in MongoDB"""
        try:
            # Extract data properly
            structured_data = extracted_data.get('structuredData', {}).get('data', {})
            
            # Store document metadata
            doc_metadata = {
                'client': structured_data.get('client', ''),
                'component': structured_data.get('component', ''),
                'certificate_number': structured_data.get('certificateNumber', ''),
                'filename': structured_data.get('filename', ''),
                'company_name': structured_data.get('companyName', ''),
                'material_specification': structured_data.get('materialSpecification', ''),
                'upload_date': datetime.now(),
                'total_heat_records': extracted_data.get('summary', {}).get('totalHeatRecords', 0),
                'validation_status': extracted_data.get('summary', {}).get('validationStatus', 'UNKNOWN'),
                'extraction_method': extracted_data.get('summary', {}).get('extractionMethod', 'openai'),
                'model_confidence': extracted_data.get('summary', {}).get('modelConfidence', 0.0),
                'raw_data': extracted_data
            }
            
            doc_id = self.documents_collection.insert_one(doc_metadata).inserted_id
            
            # Store individual heat records
            if extracted_data.get('separateRecords', {}).get('success', False):
                for record in extracted_data['separateRecords']['records']:
                    heat_record_data = record.get('heatRecord', {})
                    
                    # Extract data with proper fallbacks
                    heat_record = {
                        'document_id': doc_id,
                        'heat_no': heat_record_data.get('heatNo', ''),
                        'qty': heat_record_data.get('qty', 0),
                        
                        # Chemical composition with proper fallback
                        'chemical_composition': heat_record_data.get('chemicalComposition', {}),
                        
                        # Mechanical properties with proper fallback
                        'mechanical_properties': heat_record_data.get('mechanicalProperties', {}),
                        
                        # Hardness measurements
                        'hardness_measurements': heat_record_data.get('hardnessMeasurements', {}),
                        
                        # Base information for searching
                        'base_info': {
                            'client': record.get('client', ''),
                            'component': record.get('component', ''),
                            'certificate_number': record.get('certificateNumber', ''),
                            'company_name': record.get('companyName', ''),
                        },
                        
                        'created_at': datetime.now()
                    }
                    
                    self.heat_records_collection.insert_one(heat_record)
            
            return str(doc_id)
        except Exception as e:
            print(f"Database error: {e}")
            return None
    
    def get_all_documents(self):
        """Retrieve all documents"""
        return list(self.documents_collection.find().sort('upload_date', -1))
    
    def get_heat_records_by_document(self, doc_id):
        """Get heat records for a specific document"""
        from bson import ObjectId
        return list(self.heat_records_collection.find({'document_id': ObjectId(doc_id)}))
    
    def get_heat_record_by_heat_no(self, heat_no):
        """Get specific heat record by heat number"""
        return self.heat_records_collection.find_one({'heat_no': heat_no})
    
    def search_documents(self, query):
        """Search documents by client, component, or certificate number"""
        search_filter = {
            '$or': [
                {'client': {'$regex': query, '$options': 'i'}},
                {'component': {'$regex': query, '$options': 'i'}},
                {'certificate_number': {'$regex': query, '$options': 'i'}},
                {'company_name': {'$regex': query, '$options': 'i'}},
                {'material_specification': {'$regex': query, '$options': 'i'}}
            ]
        }
        return list(self.documents_collection.find(search_filter))