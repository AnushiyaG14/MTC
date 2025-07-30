import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import psycopg2.pool
from datetime import datetime, timezone
import json
import io
import base64
from typing import Dict, List, Optional, Any, Union
import re
try:
    import pymupdf  # Updated import for newer versions
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz  # Fallback to older import
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False
import openpyxl
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from pydantic import BaseModel, Field, validator
import openai
import requests
import os
from pathlib import Path
import tempfile
import zipfile

# Configure page
st.set_page_config(
    page_title="Steel Certificate Analysis System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# PYDANTIC MODELS
# ================================

class ChemicalComposition(BaseModel):
    carbon: Optional[float] = Field(None, ge=0, le=100, description="Carbon percentage")
    silicon: Optional[float] = Field(None, ge=0, le=100, description="Silicon percentage")
    manganese: Optional[float] = Field(None, ge=0, le=100, description="Manganese percentage")
    phosphorus: Optional[float] = Field(None, ge=0, le=100, description="Phosphorus percentage")
    sulfur: Optional[float] = Field(None, ge=0, le=100, description="Sulfur percentage")
    chromium: Optional[float] = Field(None, ge=0, le=100, description="Chromium percentage")
    nickel: Optional[float] = Field(None, ge=0, le=100, description="Nickel percentage")
    molybdenum: Optional[float] = Field(None, ge=0, le=100, description="Molybdenum percentage")
    copper: Optional[float] = Field(None, ge=0, le=100, description="Copper percentage")
    aluminum: Optional[float] = Field(None, ge=0, le=100, description="Aluminum percentage")
    nitrogen: Optional[float] = Field(None, ge=0, le=100, description="Nitrogen percentage")
    vanadium: Optional[float] = Field(None, ge=0, le=100, description="Vanadium percentage")
    titanium: Optional[float] = Field(None, ge=0, le=100, description="Titanium percentage")
    niobium: Optional[float] = Field(None, ge=0, le=100, description="Niobium percentage")
    boron: Optional[float] = Field(None, ge=0, le=100, description="Boron percentage")

class MechanicalProperties(BaseModel):
    yield_strength: Optional[float] = Field(None, ge=0, description="Yield strength in MPa")
    tensile_strength: Optional[float] = Field(None, ge=0, description="Tensile strength in MPa")
    elongation: Optional[float] = Field(None, ge=0, le=100, description="Elongation percentage")
    reduction_of_area: Optional[float] = Field(None, ge=0, le=100, description="Reduction of area percentage")
    hardness_hv: Optional[float] = Field(None, ge=0, description="Vickers hardness")
    hardness_hb: Optional[float] = Field(None, ge=0, description="Brinell hardness")
    hardness_hrc: Optional[float] = Field(None, ge=0, description="Rockwell C hardness")
    impact_energy: Optional[float] = Field(None, ge=0, description="Impact energy in Joules")
    impact_temperature: Optional[float] = Field(None, description="Impact test temperature in Celsius")

class HeatTreatment(BaseModel):
    process: Optional[str] = Field(None, description="Heat treatment process")
    temperature: Optional[float] = Field(None, description="Treatment temperature in Celsius")
    time: Optional[str] = Field(None, description="Treatment time")
    cooling_method: Optional[str] = Field(None, description="Cooling method")
    atmosphere: Optional[str] = Field(None, description="Treatment atmosphere")

class SteelCertificate(BaseModel):
    # Basic Information
    certificate_number: Optional[str] = Field(None, description="Certificate number")
    material_grade: Optional[str] = Field(None, description="Material grade")
    specification: Optional[str] = Field(None, description="Material specification")
    manufacturer: Optional[str] = Field(None, description="Manufacturer name")
    mill_name: Optional[str] = Field(None, description="Mill name")
    customer_name: Optional[str] = Field(None, description="Customer name")
    order_number: Optional[str] = Field(None, description="Order number")
    heat_number: Optional[str] = Field(None, description="Heat number")
    cast_number: Optional[str] = Field(None, description="Cast number")
    lot_number: Optional[str] = Field(None, description="Lot number")
    
    # Product Information
    product_form: Optional[str] = Field(None, description="Product form (plate, bar, etc.)")
    dimensions: Optional[str] = Field(None, description="Product dimensions")
    thickness: Optional[float] = Field(None, ge=0, description="Thickness in mm")
    width: Optional[float] = Field(None, ge=0, description="Width in mm")
    length: Optional[float] = Field(None, ge=0, description="Length in mm")
    diameter: Optional[float] = Field(None, ge=0, description="Diameter in mm")
    weight: Optional[float] = Field(None, ge=0, description="Weight in kg")
    quantity: Optional[int] = Field(None, ge=0, description="Quantity")
    
    # Dates
    manufacturing_date: Optional[str] = Field(None, description="Manufacturing date")
    test_date: Optional[str] = Field(None, description="Test date")
    certificate_date: Optional[str] = Field(None, description="Certificate date")
    
    # Composition and Properties
    chemical_composition: Optional[ChemicalComposition] = Field(None, description="Chemical composition")
    mechanical_properties: Optional[MechanicalProperties] = Field(None, description="Mechanical properties")
    heat_treatment: Optional[HeatTreatment] = Field(None, description="Heat treatment information")
    
    # Testing Information
    test_temperature: Optional[float] = Field(None, description="Test temperature in Celsius")
    test_standard: Optional[str] = Field(None, description="Test standard")
    sampling_location: Optional[str] = Field(None, description="Sampling location")
    test_direction: Optional[str] = Field(None, description="Test direction")
    
    # Certification
    certified_by: Optional[str] = Field(None, description="Certified by")
    inspector_name: Optional[str] = Field(None, description="Inspector name")
    approval_signature: Optional[str] = Field(None, description="Approval signature")
    
    # Additional
    remarks: Optional[str] = Field(None, description="Additional remarks")
    compliance_standards: Optional[List[str]] = Field(default_factory=list, description="Compliance standards")
    
    # Metadata
    file_name: Optional[str] = Field(None, description="Source file name")
    processing_timestamp: Optional[datetime] = Field(None, description="Processing timestamp")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Extraction confidence")

# ================================
# DATABASE CONNECTION (PostgreSQL)
# ================================

import psycopg2
from psycopg2.extras import RealDictCursor
import psycopg2.pool

class EnhancedDocumentDatabase:
    def __init__(self):
        self.connection_pool = None
        self.connected = False
        self.connect()
        if self.connected:
            self.create_tables()
    
    def connect(self):
        """Connect to PostgreSQL with proper error handling"""
        try:
            # Check if any PostgreSQL configuration exists
            postgres_url = st.secrets.get("POSTGRES_URL")
            postgres_host = st.secrets.get("POSTGRES_HOST")
            
            if not postgres_url and not postgres_host:
                st.error("🔗 **PostgreSQL Configuration Missing!**")
                st.info("""
                **Please add PostgreSQL configuration to your Streamlit secrets:**
                
                **Option 1 - Connection String (Recommended):**
                ```toml
                POSTGRES_URL = "postgresql://username:password@host:port/database"
                ```
                
                **Option 2 - Individual Parameters:**
                ```toml
                POSTGRES_HOST = "your-host"
                POSTGRES_DATABASE = "steel_certificates"
                POSTGRES_USER = "your-username"
                POSTGRES_PASSWORD = "your-password"
                POSTGRES_PORT = 5432
                POSTGRES_SSLMODE = "require"
                ```
                
                **Free PostgreSQL Options:**
                - 🌟 **Supabase**: supabase.com (500MB free)
                - 🚂 **Railway**: railway.app (1GB free)
                - ⚡ **Neon**: neon.tech (512MB free)
                """)
                return False
            
            # PostgreSQL connection parameters
            if postgres_url:
                # Use connection string directly
                self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                    1, 20, postgres_url
                )
            else:
                # Use individual parameters
                db_params = {
                    'host': st.secrets.get("POSTGRES_HOST"),
                    'database': st.secrets.get("POSTGRES_DATABASE", "steel_certificates"),
                    'user': st.secrets.get("POSTGRES_USER"),
                    'password': st.secrets.get("POSTGRES_PASSWORD"),
                    'port': st.secrets.get("POSTGRES_PORT", 5432),
                    'sslmode': st.secrets.get("POSTGRES_SSLMODE", "prefer")
                }
                
                # Validate required parameters
                if not all([db_params['host'], db_params['user'], db_params['password']]):
                    st.error("❌ Missing required PostgreSQL parameters (host, user, password)")
                    return False
                
                self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                    1, 20, **db_params
                )
            
            # Test connection
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            self.connection_pool.putconn(conn)
            
            self.connected = True
            return True
            
        except psycopg2.OperationalError as e:
            error_msg = str(e)
            if "Connection refused" in error_msg:
                st.error("🔗 **PostgreSQL Connection Refused**")
                st.info("""
                **Common causes:**
                1. **Wrong host/port** - Check your database URL
                2. **Database not running** - Verify your PostgreSQL service is active
                3. **Firewall blocking** - Ensure port 5432 is accessible
                4. **SSL issues** - Try adding `?sslmode=require` to your URL
                
                **Quick fixes:**
                - For Supabase: Use the connection pooler URL (port 6543)
                - For Railway: Get the public URL from the Railway dashboard
                - For local testing: Start PostgreSQL service
                """)
            elif "authentication failed" in error_msg:
                st.error("🔐 **PostgreSQL Authentication Failed**")
                st.info("Check your username and password in the secrets configuration.")
            elif "does not exist" in error_msg:
                st.error("🗄️ **Database Does Not Exist**")
                st.info("Verify the database name in your connection string.")
            else:
                st.error(f"🔗 **PostgreSQL Connection Error:** {error_msg}")
            
            return False
            
        except Exception as e:
            st.error(f"🔗 **Database Connection Failed:** {str(e)}")
            return False
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        if not self.connected or not self.connection_pool:
            return
            
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255) UNIQUE,
                    file_name VARCHAR(255),
                    certificate_number VARCHAR(100),
                    heat_number VARCHAR(100),
                    material_grade VARCHAR(100),
                    specification VARCHAR(255),
                    manufacturer VARCHAR(255),
                    customer_name VARCHAR(255),
                    order_number VARCHAR(100),
                    cast_number VARCHAR(100),
                    lot_number VARCHAR(100),
                    product_form VARCHAR(100),
                    dimensions VARCHAR(255),
                    thickness FLOAT,
                    width FLOAT,
                    length FLOAT,
                    diameter FLOAT,
                    weight FLOAT,
                    quantity INTEGER,
                    manufacturing_date VARCHAR(50),
                    test_date VARCHAR(50),
                    certificate_date VARCHAR(50),
                    chemical_composition JSONB,
                    mechanical_properties JSONB,
                    heat_treatment JSONB,
                    test_temperature FLOAT,
                    test_standard VARCHAR(255),
                    sampling_location VARCHAR(255),
                    test_direction VARCHAR(100),
                    certified_by VARCHAR(255),
                    inspector_name VARCHAR(255),
                    approval_signature VARCHAR(255),
                    remarks TEXT,
                    compliance_standards JSONB,
                    confidence_score FLOAT,
                    file_type VARCHAR(50),
                    extracted_text_length INTEGER,
                    processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    raw_data JSONB
                );
                
                CREATE INDEX IF NOT EXISTS idx_documents_cert_num ON documents(certificate_number);
                CREATE INDEX IF NOT EXISTS idx_documents_heat_num ON documents(heat_number);
                CREATE INDEX IF NOT EXISTS idx_documents_timestamp ON documents(processing_timestamp);
            """)
            
            # Training data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id SERIAL PRIMARY KEY,
                    field_name VARCHAR(100),
                    extracted_value TEXT,
                    correct_value TEXT,
                    context TEXT,
                    is_correct BOOLEAN,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_training_field ON training_data(field_name);
                CREATE INDEX IF NOT EXISTS idx_training_timestamp ON training_data(timestamp);
            """)
            
            conn.commit()
            cursor.close()
            self.connection_pool.putconn(conn)
            
        except Exception as e:
            st.error(f"Error creating tables: {str(e)}")
    
    def store_document(self, document_data: Dict) -> str:
        """Store a processed document"""
        if not self.connected or not self.connection_pool:
            st.warning("⚠️ Database not connected. Document not saved.")
            return None
            
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            # Generate unique document ID
            document_id = f"{document_data.get('file_name', 'unknown')}_{datetime.now().isoformat()}"
            
            # Helper function to safely serialize JSON data
            def safe_json_dumps(data):
                if data is None:
                    return None
                try:
                    return json.dumps(data, default=str)  # Convert non-serializable objects to strings
                except Exception:
                    return json.dumps(str(data))  # Fallback to string conversion
            
            # Extract fields for PostgreSQL storage
            insert_query = """
                INSERT INTO documents (
                    document_id, file_name, certificate_number, heat_number, 
                    material_grade, specification, manufacturer, customer_name,
                    order_number, cast_number, lot_number, product_form, dimensions,
                    thickness, width, length, diameter, weight, quantity,
                    manufacturing_date, test_date, certificate_date,
                    chemical_composition, mechanical_properties, heat_treatment,
                    test_temperature, test_standard, sampling_location, test_direction,
                    certified_by, inspector_name, approval_signature, remarks,
                    compliance_standards, confidence_score, file_type,
                    extracted_text_length, raw_data
                ) VALUES (
                    %(document_id)s, %(file_name)s, %(certificate_number)s, %(heat_number)s, 
                    %(material_grade)s, %(specification)s, %(manufacturer)s, %(customer_name)s,
                    %(order_number)s, %(cast_number)s, %(lot_number)s, %(product_form)s, 
                    %(dimensions)s, %(thickness)s, %(width)s, %(length)s, %(diameter)s, 
                    %(weight)s, %(quantity)s, %(manufacturing_date)s, %(test_date)s, 
                    %(certificate_date)s, %(chemical_composition)s, %(mechanical_properties)s, 
                    %(heat_treatment)s, %(test_temperature)s, %(test_standard)s, 
                    %(sampling_location)s, %(test_direction)s, %(certified_by)s, 
                    %(inspector_name)s, %(approval_signature)s, %(remarks)s,
                    %(compliance_standards)s, %(confidence_score)s, %(file_type)s,
                    %(extracted_text_length)s, %(raw_data)s
                )
            """
            
            # Prepare data dictionary for named parameters
            data_params = {
                'document_id': document_id,
                'file_name': document_data.get('file_name'),
                'certificate_number': document_data.get('certificate_number'),
                'heat_number': document_data.get('heat_number'),
                'material_grade': document_data.get('material_grade'),
                'specification': document_data.get('specification'),
                'manufacturer': document_data.get('manufacturer'),
                'customer_name': document_data.get('customer_name'),
                'order_number': document_data.get('order_number'),
                'cast_number': document_data.get('cast_number'),
                'lot_number': document_data.get('lot_number'),
                'product_form': document_data.get('product_form'),
                'dimensions': document_data.get('dimensions'),
                'thickness': document_data.get('thickness'),
                'width': document_data.get('width'),
                'length': document_data.get('length'),
                'diameter': document_data.get('diameter'),
                'weight': document_data.get('weight'),
                'quantity': document_data.get('quantity'),
                'manufacturing_date': document_data.get('manufacturing_date'),
                'test_date': document_data.get('test_date'),
                'certificate_date': document_data.get('certificate_date'),
                'chemical_composition': safe_json_dumps(document_data.get('chemical_composition')),
                'mechanical_properties': safe_json_dumps(document_data.get('mechanical_properties')),
                'heat_treatment': safe_json_dumps(document_data.get('heat_treatment')),
                'test_temperature': document_data.get('test_temperature'),
                'test_standard': document_data.get('test_standard'),
                'sampling_location': document_data.get('sampling_location'),
                'test_direction': document_data.get('test_direction'),
                'certified_by': document_data.get('certified_by'),
                'inspector_name': document_data.get('inspector_name'),
                'approval_signature': document_data.get('approval_signature'),
                'remarks': document_data.get('remarks'),
                'compliance_standards': safe_json_dumps(document_data.get('compliance_standards', [])),
                'confidence_score': document_data.get('confidence_score'),
                'file_type': document_data.get('file_type'),
                'extracted_text_length': document_data.get('extracted_text_length'),
                'raw_data': safe_json_dumps(document_data)
            }
            
            cursor.execute(insert_query, data_params)
            
            conn.commit()
            cursor.close()
            self.connection_pool.putconn(conn)
            
            return document_id
            
        except Exception as e:
            st.error(f"Error storing document: {str(e)}")
            return None
    
    def get_documents(self, limit: int = 100) -> List[Dict]:
        """Retrieve stored documents"""
        if not self.connected or not self.connection_pool:
            return []
            
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM documents 
                ORDER BY processing_timestamp DESC 
                LIMIT %s
            """, (limit,))
            
            results = cursor.fetchall()
            
            # Convert to list of dicts and parse JSON fields
            documents = []
            for row in results:
                doc = dict(row)
                # Parse JSON fields
                if doc.get('chemical_composition'):
                    doc['chemical_composition'] = json.loads(doc['chemical_composition'])
                if doc.get('mechanical_properties'):
                    doc['mechanical_properties'] = json.loads(doc['mechanical_properties'])
                if doc.get('heat_treatment'):
                    doc['heat_treatment'] = json.loads(doc['heat_treatment'])
                if doc.get('compliance_standards'):
                    doc['compliance_standards'] = json.loads(doc['compliance_standards'])
                if doc.get('raw_data'):
                    doc['raw_data'] = json.loads(doc['raw_data'])
                documents.append(doc)
            
            cursor.close()
            self.connection_pool.putconn(conn)
            
            return documents
            
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def search_documents(self, search_params: Dict) -> List[Dict]:
        """Search documents with filters"""
        if not self.connected or not self.connection_pool:
            return []
            
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build WHERE clause based on search parameters
            where_conditions = []
            params = []
            
            for field, value in search_params.items():
                if field == "$or":
                    # Handle OR conditions
                    or_conditions = []
                    for or_condition in value:
                        for or_field, or_value in or_condition.items():
                            if "$regex" in or_value:
                                or_conditions.append(f"{or_field} ILIKE %s")
                                params.append(f"%{or_value['$regex']}%")
                    if or_conditions:
                        where_conditions.append(f"({' OR '.join(or_conditions)})")
                elif isinstance(value, dict) and "$regex" in value:
                    where_conditions.append(f"{field} ILIKE %s")
                    params.append(f"%{value['$regex']}%")
                else:
                    where_conditions.append(f"{field} = %s")
                    params.append(value)
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            query = f"""
                SELECT * FROM documents 
                WHERE {where_clause}
                ORDER BY processing_timestamp DESC
            """
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            # Convert to list of dicts and parse JSON fields
            documents = []
            for row in results:
                doc = dict(row)
                # Parse JSON fields
                if doc.get('chemical_composition'):
                    doc['chemical_composition'] = json.loads(doc['chemical_composition'])
                if doc.get('mechanical_properties'):
                    doc['mechanical_properties'] = json.loads(doc['mechanical_properties'])
                if doc.get('heat_treatment'):
                    doc['heat_treatment'] = json.loads(doc['heat_treatment'])
                if doc.get('compliance_standards'):
                    doc['compliance_standards'] = json.loads(doc['compliance_standards'])
                documents.append(doc)
            
            cursor.close()
            self.connection_pool.putconn(conn)
            
            return documents
            
        except Exception as e:
            st.error(f"Error searching documents: {str(e)}")
            return []
    
    def store_training_data(self, field_name: str, extracted_value: str, correct_value: str, context: str):
        """Store training data for ML model improvement"""
        if not self.connected or not self.connection_pool:
            st.warning("⚠️ Database not connected. Training data not saved.")
            return False
            
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO training_data 
                (field_name, extracted_value, correct_value, context, is_correct)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                field_name,
                extracted_value,
                correct_value,
                context,
                extracted_value == correct_value
            ))
            
            conn.commit()
            cursor.close()
            self.connection_pool.putconn(conn)
            
            return True
            
        except Exception as e:
            st.error(f"Error storing training data: {str(e)}")
            return False
    
    def get_training_data(self) -> List[Dict]:
        """Retrieve training data"""
        if not self.connected or not self.connection_pool:
            return []
            
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM training_data 
                ORDER BY timestamp DESC
            """)
            
            results = cursor.fetchall()
            training_data = [dict(row) for row in results]
            
            cursor.close()
            self.connection_pool.putconn(conn)
            
            return training_data
            
        except Exception as e:
            st.error(f"Error retrieving training data: {str(e)}")
            return []

# ================================
# ADVANCED DOCUMENT PARSER
# ================================

class AdvancedDocumentParser:
    def __init__(self):
        self.openai_client = None
        self.field_classifiers = {}
        self.initialize_openai()
        self.load_trained_models()
    
    def initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
            if api_key:
                openai.api_key = api_key
                self.openai_client = openai
        except Exception as e:
            st.warning(f"OpenAI initialization failed: {str(e)}")
    
    def load_trained_models(self):
        """Load trained ML models"""
        try:
            # Try to load saved models from session state
            if 'trained_models' in st.session_state:
                self.field_classifiers = st.session_state.trained_models
        except Exception as e:
            st.info("No pre-trained models found. Will train on first use.")
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF with multiple fallback methods"""
        try:
            if not PYMUPDF_AVAILABLE:
                return "PDF extraction requires PyMuPDF. Please install: pip install pymupdf"
            
            # Try newer PyMuPDF API first
            try:
                if 'pymupdf' in globals():
                    doc = pymupdf.open(stream=file_content, filetype="pdf")
                else:
                    doc = fitz.open(stream=file_content, filetype="pdf")
            except Exception as e:
                # Try alternative opening methods
                try:
                    # Method 1: Open from bytes directly
                    if 'pymupdf' in globals():
                        doc = pymupdf.open("pdf", file_content)
                    else:
                        doc = fitz.open("pdf", file_content)
                except Exception as e2:
                    # Method 2: Save to temp file and open
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                            tmp_file.write(file_content)
                            tmp_file.flush()
                            
                            if 'pymupdf' in globals():
                                doc = pymupdf.open(tmp_file.name)
                            else:
                                doc = fitz.open(tmp_file.name)
                        
                        # Clean up temp file
                        os.unlink(tmp_file.name)
                    except Exception as e3:
                        st.error(f"PDF opening failed with all methods: {str(e)}, {str(e2)}, {str(e3)}")
                        return self._extract_pdf_with_alternative_methods(file_content)
            
            # Extract text from all pages
            text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text() + "\n"
            
            doc.close()
            
            if not text.strip():
                # If no text extracted, try alternative methods
                return self._extract_pdf_with_alternative_methods(file_content)
            
            return text
            
        except Exception as e:
            st.warning(f"PyMuPDF extraction failed: {str(e)}")
            return self._extract_pdf_with_alternative_methods(file_content)
    
    def _extract_pdf_with_alternative_methods(self, file_content: bytes) -> str:
        """Alternative PDF extraction methods"""
        try:
            # Method 1: Try with PyPDF2 if available
            try:
                import PyPDF2
                pdf_file = io.BytesIO(file_content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                if text.strip():
                    return text
            except ImportError:
                pass
            except Exception as e:
                st.warning(f"PyPDF2 extraction failed: {str(e)}")
            
            # Method 2: Try with pdfplumber if available
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                if text.strip():
                    return text
            except ImportError:
                pass
            except Exception as e:
                st.warning(f"pdfplumber extraction failed: {str(e)}")
            
            # Method 3: Basic text extraction attempt
            try:
                # Try to decode as text (might work for simple PDFs)
                text = file_content.decode('utf-8', errors='ignore')
                if len(text) > 100 and 'PDF' not in text[:50]:  # Basic validation
                    return text
            except:
                pass
            
            # If all methods fail, return informative message
            return """PDF text extraction failed with all available methods. 

To fix this, you can:
1. Install PyMuPDF: pip install pymupdf
2. Or install PyPDF2: pip install PyPDF2
3. Or install pdfplumber: pip install pdfplumber
4. Or convert your PDF to text/Word format and re-upload

Alternative: Copy the text from your PDF and paste it into a .txt file to upload."""
            
        except Exception as e:
            return f"PDF text extraction completely failed: {str(e)}"
    
    def extract_text_from_excel(self, file_content: bytes) -> str:
        """Extract text from Excel"""
        try:
            df = pd.read_excel(io.BytesIO(file_content))
            return df.to_string()
        except Exception as e:
            st.error(f"Excel text extraction failed: {str(e)}")
            return ""
    
    def extract_text_from_image(self, file_content: bytes) -> str:
        """Extract text from image using OCR (placeholder)"""
        try:
            # This would require pytesseract for full OCR
            return "Image text extraction requires OCR setup (pytesseract)"
        except Exception as e:
            st.error(f"Image text extraction failed: {str(e)}")
            return ""
    
    def intelligent_field_extraction(self, text: str) -> Dict:
        """Advanced intelligent field extraction using multiple patterns"""
        result = {}
        
        # Enhanced regex patterns for various certificate formats
        advanced_patterns = {
            'certificate_number': [
                r'(?i)certificate\s*(?:no|number|#)\.?\s*:?\s*([A-Z0-9\-/\.]+)',
                r'(?i)cert\.?\s*(?:no|#|number)\.?\s*:?\s*([A-Z0-9\-/\.]+)',
                r'(?i)document\s*(?:no|number)\.?\s*:?\s*([A-Z0-9\-/\.]+)',
                r'(?i)serial\s*(?:no|number)\.?\s*:?\s*([A-Z0-9\-/\.]+)'
            ],
            'heat_number': [
                r'(?i)heat\s*(?:no|number)\.?\s*:?\s*([A-Z0-9\-/\.]+)',
                r'(?i)heat\s*([A-Z0-9\-/\.]+)',
                r'(?i)ladle\s*(?:no|number)\.?\s*:?\s*([A-Z0-9\-/\.]+)',
                r'(?i)melt\s*(?:no|number)\.?\s*:?\s*([A-Z0-9\-/\.]+)'
            ],
            'material_grade': [
                r'(?i)grade\s*:?\s*([A-Z0-9\-/\.]+)',
                r'(?i)material\s*grade\s*:?\s*([A-Z0-9\-/\.]+)',
                r'(?i)steel\s*grade\s*:?\s*([A-Z0-9\-/\.]+)',
                r'(?i)alloy\s*:?\s*([A-Z0-9\-/\.]+)'
            ],
            'specification': [
                r'(?i)specification\s*:?\s*([A-Z0-9\s\-/\.]+)',
                r'(?i)spec\.?\s*:?\s*([A-Z0-9\s\-/\.]+)',
                r'(?i)standard\s*:?\s*([A-Z0-9\s\-/\.]+)',
                r'(?i)norm\s*:?\s*([A-Z0-9\s\-/\.]+)'
            ],
            'manufacturer': [
                r'(?i)manufacturer\s*:?\s*([A-Za-z0-9\s\-&\.]+)',
                r'(?i)producer\s*:?\s*([A-Za-z0-9\s\-&\.]+)',
                r'(?i)mill\s*:?\s*([A-Za-z0-9\s\-&\.]+)',
                r'(?i)supplier\s*:?\s*([A-Za-z0-9\s\-&\.]+)'
            ]
        }
        
        # Chemical composition patterns
        chemical_patterns = {
            'carbon': [r'(?i)c\s*[:\-]?\s*(\d+\.?\d*)', r'(?i)carbon\s*[:\-]?\s*(\d+\.?\d*)'],
            'silicon': [r'(?i)si\s*[:\-]?\s*(\d+\.?\d*)', r'(?i)silicon\s*[:\-]?\s*(\d+\.?\d*)'],
            'manganese': [r'(?i)mn\s*[:\-]?\s*(\d+\.?\d*)', r'(?i)manganese\s*[:\-]?\s*(\d+\.?\d*)'],
            'phosphorus': [r'(?i)p\s*[:\-]?\s*(\d+\.?\d*)', r'(?i)phosphorus\s*[:\-]?\s*(\d+\.?\d*)'],
            'sulfur': [r'(?i)s\s*[:\-]?\s*(\d+\.?\d*)', r'(?i)sulfur\s*[:\-]?\s*(\d+\.?\d*)'],
            'chromium': [r'(?i)cr\s*[:\-]?\s*(\d+\.?\d*)', r'(?i)chromium\s*[:\-]?\s*(\d+\.?\d*)'],
            'nickel': [r'(?i)ni\s*[:\-]?\s*(\d+\.?\d*)', r'(?i)nickel\s*[:\-]?\s*(\d+\.?\d*)'],
            'molybdenum': [r'(?i)mo\s*[:\-]?\s*(\d+\.?\d*)', r'(?i)molybdenum\s*[:\-]?\s*(\d+\.?\d*)']
        }
        
        # Mechanical properties patterns
        mechanical_patterns = {
            'yield_strength': [
                r'(?i)yield\s*strength\s*[:\-]?\s*(\d+\.?\d*)',
                r'(?i)ys\s*[:\-]?\s*(\d+\.?\d*)',
                r'(?i)rp0?\.?2\s*[:\-]?\s*(\d+\.?\d*)'
            ],
            'tensile_strength': [
                r'(?i)tensile\s*strength\s*[:\-]?\s*(\d+\.?\d*)',
                r'(?i)uts\s*[:\-]?\s*(\d+\.?\d*)',
                r'(?i)rm\s*[:\-]?\s*(\d+\.?\d*)'
            ],
            'elongation': [
                r'(?i)elongation\s*[:\-]?\s*(\d+\.?\d*)',
                r'(?i)el\s*[:\-]?\s*(\d+\.?\d*)',
                r'(?i)a\s*[:\-]?\s*(\d+\.?\d*)\s*%'
            ]
        }
        
        # Apply basic information patterns
        for field, field_patterns in advanced_patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result[field] = match.group(1).strip()
                    break
        
        # Chemical composition
        chemical_composition = {}
        for element, element_patterns in chemical_patterns.items():
            for pattern in element_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        chemical_composition[element] = float(match.group(1))
                        break
                    except ValueError:
                        continue
        
        if chemical_composition:
            result['chemical_composition'] = chemical_composition
        
        # Mechanical properties
        mechanical_properties = {}
        for prop, prop_patterns in mechanical_patterns.items():
            for pattern in prop_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        mechanical_properties[prop] = float(match.group(1))
                        break
                    except ValueError:
                        continue
        
        if mechanical_properties:
            result['mechanical_properties'] = mechanical_properties
        
        return result
    
    def parse_with_openai(self, text: str, filename: str) -> Dict:
        """Parse document using OpenAI GPT-4 with improved error handling"""
        if not self.openai_client:
            return self.intelligent_field_extraction(text)
        
        try:
            prompt = f"""
            Extract steel certificate information from this text and return ONLY valid JSON.
            
            Extract these fields if available:
            - certificate_number, heat_number, material_grade, specification
            - manufacturer, customer_name, order_number
            - chemical_composition (as object with elements like carbon, silicon, etc.)
            - mechanical_properties (as object with yield_strength, tensile_strength, etc.)
            - heat_treatment (as object with process, temperature, etc.)
            
            Return ONLY a valid JSON object, no additional text or explanations.
            
            Example format:
            {{
                "certificate_number": "ABC123",
                "heat_number": "H456",
                "material_grade": "S355",
                "chemical_composition": {{"carbon": 0.15, "silicon": 0.3}},
                "mechanical_properties": {{"yield_strength": 355, "tensile_strength": 510}}
            }}
            
            Text to analyze:
            {text[:3000]}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a steel certificate parser. Return only valid JSON with extracted data. No additional text or explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean up response text - remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Try to parse JSON
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as json_error:
                st.warning(f"OpenAI returned invalid JSON: {str(json_error)}")
                st.info("Falling back to intelligent extraction...")
                return self.intelligent_field_extraction(text)
            
            # Add metadata
            result['file_name'] = filename
            result['confidence_score'] = 0.9
            
            return result
            
        except Exception as e:
            st.warning(f"OpenAI parsing failed: {str(e)}")
            st.info("Using intelligent extraction instead...")
            fallback_result = self.intelligent_field_extraction(text)
            fallback_result['file_name'] = filename
            fallback_result['confidence_score'] = 0.6
            return fallback_result
    
    def process_document(self, file_content: bytes, file_type: str, filename: str) -> Dict:
        """Main document processing function with improved error handling"""
        try:
            # Extract text based on file type
            if file_type == "pdf" or filename.lower().endswith('.pdf'):
                text = self.extract_text_from_pdf(file_content)
            elif file_type in ["xlsx", "xls"] or filename.lower().endswith(('.xlsx', '.xls')):
                text = self.extract_text_from_excel(file_content)
            elif file_type in ["jpg", "jpeg", "png", "tiff"] or filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
                text = self.extract_text_from_image(file_content)
            else:
                try:
                    text = file_content.decode('utf-8', errors='ignore')
                except:
                    text = str(file_content)
            
            if not text.strip():
                return {'error': 'No text could be extracted from the document'}
            
            # Parse with available method
            result = self.parse_with_openai(text, filename)
            
            # Add processing metadata with proper serialization
            result['processing_timestamp'] = datetime.now(timezone.utc).isoformat()  # Convert to string
            result['extracted_text_length'] = len(text)
            result['file_type'] = file_type
            
            return result
            
        except Exception as e:
            return {'error': f'Document processing failed: {str(e)}'}
    
    def train_field_classifier(self, field_name: str, training_data: List[Dict]):
        """Train ML classifier for specific field"""
        try:
            if len(training_data) < 10:
                st.warning(f"Need at least 10 training samples for {field_name}")
                return False
            
            # Prepare training data
            texts = [item['context'] for item in training_data]
            labels = [item['correct_value'] for item in training_data]
            
            # Feature extraction
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X = vectorizer.fit_transform(texts)
            
            # Encode labels
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(labels)
            
            # Train classifier
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            classifier.fit(X, y)
            
            # Save model
            model_data = {
                'classifier': classifier,
                'vectorizer': vectorizer,
                'label_encoder': label_encoder
            }
            
            # Store in session state
            if 'trained_models' not in st.session_state:
                st.session_state.trained_models = {}
            
            st.session_state.trained_models[field_name] = model_data
            self.field_classifiers[field_name] = model_data
            
            return True
            
        except Exception as e:
            st.error(f"Training failed for {field_name}: {str(e)}")
            return False

# ================================
# UTILITY FUNCTIONS
# ================================

def initialize_session_state():
    """Initialize session state variables"""
    if 'db' not in st.session_state:
        st.session_state.db = EnhancedDocumentDatabase()
    
    if 'parser' not in st.session_state:
        st.session_state.parser = AdvancedDocumentParser()
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "dashboard"

def create_csv_from_result(result):
    """Create CSV data from extraction result"""
    rows = []
    
    # Basic info
    rows.append({
        'Type': 'Basic Information',
        'Field': 'Certificate Number',
        'Value': result.get('certificate_number', ''),
        'Unit': '',
        'Category': 'Document'
    })
    
    # Chemical composition
    if result.get('chemical_composition'):
        for element, percentage in result['chemical_composition'].items():
            if percentage is not None:
                rows.append({
                    'Type': 'Chemical Composition',
                    'Field': element.upper(),
                    'Value': percentage,
                    'Unit': '%',
                    'Category': 'Chemistry'
                })
    
    # Mechanical properties
    if result.get('mechanical_properties'):
        for prop, value in result['mechanical_properties'].items():
            if value is not None:
                unit = ""
                if 'strength' in prop.lower():
                    unit = "MPa"
                elif 'elongation' in prop.lower():
                    unit = "%"
                elif 'hardness' in prop.lower():
                    unit = "HV/HB/HRC"
                
                rows.append({
                    'Type': 'Mechanical Properties',
                    'Field': prop.replace('_', ' ').title(),
                    'Value': value,
                    'Unit': unit,
                    'Category': 'Mechanical'
                })
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

def generate_certificate_report(result, filename):
    """Generate HTML certificate report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Steel Certificate Report - {filename}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: linear-gradient(135deg, #1f77b4, #17a2b8); color: white; padding: 20px; border-radius: 8px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔬 Steel Certificate Analysis Report</h1>
            <p><strong>File:</strong> {filename}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📋 Certificate Information</h2>
            <p><strong>Certificate Number:</strong> {result.get('certificate_number', 'N/A')}</p>
            <p><strong>Heat Number:</strong> {result.get('heat_number', 'N/A')}</p>
            <p><strong>Material Grade:</strong> {result.get('material_grade', 'N/A')}</p>
            <p><strong>Manufacturer:</strong> {result.get('manufacturer', 'N/A')}</p>
        </div>
    </body>
    </html>
    """
    return html_content

def generate_analytics_report(documents):
    """Generate HTML analytics report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Steel Certificate Analytics Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #1f77b4; color: white; padding: 20px; text-align: center; }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #f0f2f6; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Steel Certificate Analytics Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Summary Statistics</h2>
            <div class="metric">
                <h3>{len(documents)}</h3>
                <p>Total Documents</p>
            </div>
            <div class="metric">
                <h3>{len(set(doc.get('heat_number') for doc in documents if doc.get('heat_number')))}</h3>
                <p>Unique Heat Numbers</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# ================================
# STREAMLIT PAGES
# ================================

def main_dashboard():
    """Main dashboard page"""
    st.markdown('<h1 class="main-header">🔬 Steel Certificate Analysis System</h1>', unsafe_allow_html=True)
    
    # Get statistics
    documents = st.session_state.db.get_documents(limit=1000)
    training_data = st.session_state.db.get_training_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Documents", len(documents))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        processed_today = len([d for d in documents if d.get('processing_timestamp', datetime.min).date() == datetime.now().date()])
        st.metric("Processed Today", processed_today)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        unique_heats = len(set(d.get('heat_number') for d in documents if d.get('heat_number')))
        st.metric("Unique Heat Numbers", unique_heats)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Training Records", len(training_data))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent activity
    st.subheader("📈 Processing Activity")
    
    if documents:
        df = pd.DataFrame(documents)
        if 'processing_timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['processing_timestamp']).dt.date
            daily_counts = df.groupby('date').size().reset_index(name='count')
            
            fig = px.line(daily_counts, x='date', y='count', 
                         title='Daily Document Processing')
            st.plotly_chart(fig, use_container_width=True)

def upload_documents_page():
    """Document upload and processing page"""
    st.title("📤 Upload Documents")
    
    st.markdown("""
    Upload steel certificates in PDF, Excel, or image format for automated analysis.
    The system will extract key information including chemical composition, mechanical properties, and certification details.
    """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['pdf', 'xlsx', 'xls', 'jpg', 'jpeg', 'png', 'tiff', 'txt'],
        help="Upload multiple files for batch processing"
    )
    
    if uploaded_files:
        st.subheader(f"🔄 Processing {len(uploaded_files)} file(s)")
        
        progress_bar = st.progress(0)
        results = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            st.info(f"📄 Processing: {uploaded_file.name}")
            
            try:
                file_content = uploaded_file.read()
                file_type = uploaded_file.type or 'application/octet-stream'
                
                # Process document
                with st.spinner(f"Analyzing {uploaded_file.name}..."):
                    result = st.session_state.parser.process_document(
                        file_content, file_type, uploaded_file.name
                    )
                
                # Validate with Pydantic
                if 'error' not in result:
                    try:
                        steel_cert = SteelCertificate(**result)
                        result['validation_passed'] = True
                    except Exception as e:
                        result['validation_passed'] = False
                        result['validation_error'] = str(e)
                    
                    # Store in database
                    document_id = st.session_state.db.store_document(result)
                    result['document_id'] = document_id
                
                results.append({
                    'file_name': uploaded_file.name,
                    'result': result
                })
                
            except Exception as e:
                results.append({
                    'file_name': uploaded_file.name,
                    'result': {'error': f'Processing failed: {str(e)}'}
                })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Display results
        st.subheader("📋 Processing Results")
        
        for i, result_data in enumerate(results):
            file_name = result_data['file_name']
            result = result_data['result']
            
            with st.expander(f"📄 {file_name}", expanded=True):
                if 'error' in result:
                    st.error(f"❌ Error: {result['error']}")
                else:
                    if result.get('validation_passed'):
                        st.success("✅ Document processed and validated successfully!")
                    else:
                        st.warning(f"⚠️ Processed with validation issues")
                    
                    # Display extracted information
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Basic Information:**")
                        st.write(f"• Certificate Number: {result.get('certificate_number', 'N/A')}")
                        st.write(f"• Heat Number: {result.get('heat_number', 'N/A')}")
                        st.write(f"• Material Grade: {result.get('material_grade', 'N/A')}")
                        st.write(f"• Confidence Score: {result.get('confidence_score', 0):.2%}")
                    
                    with col2:
                        st.write("**Product Information:**")
                        st.write(f"• Manufacturer: {result.get('manufacturer', 'N/A')}")
                        st.write(f"• Specification: {result.get('specification', 'N/A')}")
                        st.write(f"• Customer: {result.get('customer_name', 'N/A')}")
                    
                    # Chemical composition
                    if result.get('chemical_composition'):
                        st.write("**Chemical Composition (%):**")
                        comp_data = []
                        for element, value in result['chemical_composition'].items():
                            if value is not None:
                                comp_data.append({'Element': element.title(), 'Percentage': value})
                        
                        if comp_data:
                            df_comp = pd.DataFrame(comp_data)
                            st.dataframe(df_comp, use_container_width=True)
                    
                    # Download options
                    json_data = json.dumps(result, indent=2, default=str)
                    st.download_button(
                        "📄 Download JSON",
                        json_data,
                        f"{file_name}_extracted.json",
                        "application/json",
                        key=f"json_{i}"
                    )

def analytics_page():
    """Analytics and visualization page"""
    st.title("📊 Analytics Dashboard")
    
    documents = st.session_state.db.get_documents(limit=1000)
    
    if not documents:
        st.info("No documents found. Please upload some documents first.")
        return
    
    df = pd.DataFrame(documents)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(df))
    
    with col2:
        unique_heats = df['heat_number'].nunique() if 'heat_number' in df.columns else 0
        st.metric("Unique Heats", unique_heats)
    
    with col3:
        unique_grades = df['material_grade'].nunique() if 'material_grade' in df.columns else 0
        st.metric("Material Grades", unique_grades)
    
    with col4:
        avg_confidence = df['confidence_score'].mean() if 'confidence_score' in df.columns else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    # Charts
    if 'material_grade' in df.columns:
        st.subheader("🏭 Material Grade Distribution")
        grade_counts = df['material_grade'].value_counts().head(10)
        fig = px.bar(x=grade_counts.index, y=grade_counts.values,
                    title='Top Material Grades')
        st.plotly_chart(fig, use_container_width=True)

def document_library_page():
    """Document library and search page"""
    st.title("📚 Document Library")
    
    # Search interface
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_field = st.selectbox(
            "Search by",
            ["All Fields", "Certificate Number", "Heat Number", "Material Grade", "Manufacturer"]
        )
    
    with col2:
        search_value = st.text_input("Search value")
    
    with col3:
        if st.button("🔍 Search", use_container_width=True):
            st.rerun()
    
    # Get documents
    if search_value:
        if search_field == "All Fields":
            search_query = {
                "$or": [
                    {"certificate_number": {"$regex": search_value, "$options": "i"}},
                    {"heat_number": {"$regex": search_value, "$options": "i"}},
                    {"material_grade": {"$regex": search_value, "$options": "i"}},
                    {"manufacturer": {"$regex": search_value, "$options": "i"}}
                ]
            }
        else:
            field_map = {
                "Certificate Number": "certificate_number",
                "Heat Number": "heat_number", 
                "Material Grade": "material_grade",
                "Manufacturer": "manufacturer"
            }
            field_name = field_map[search_field]
            search_query = {field_name: {"$regex": search_value, "$options": "i"}}
        
        documents = st.session_state.db.search_documents(search_query)
    else:
        documents = st.session_state.db.get_documents(limit=100)
    
    st.subheader(f"📄 Documents ({len(documents)} found)")
    
    if documents:
        for i, doc in enumerate(documents):
            with st.expander(f"📄 {doc.get('file_name', f'Document {i+1}')}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Information:**")
                    st.write(f"• Certificate Number: {doc.get('certificate_number', 'N/A')}")
                    st.write(f"• Heat Number: {doc.get('heat_number', 'N/A')}")
                    st.write(f"• Material Grade: {doc.get('material_grade', 'N/A')}")
                    st.write(f"• Manufacturer: {doc.get('manufacturer', 'N/A')}")
                
                with col2:
                    st.write("**Processing Info:**")
                    if doc.get('processing_timestamp'):
                        st.write(f"• Processed: {doc['processing_timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"• Confidence: {doc.get('confidence_score', 0):.2%}")
                
                # Export button
                json_str = json.dumps(doc, indent=2, default=str)
                st.download_button(
                    label="💾 Export JSON",
                    data=json_str,
                    file_name=f"{doc.get('file_name', 'document')}.json",
                    mime="application/json",
                    key=f"download_{i}"
                )
        
        # Bulk export
        st.subheader("📦 Bulk Export")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export All as JSON"):
                json_data = json.dumps(documents, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="steel_certificates.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Export as CSV"):
                df = pd.DataFrame(documents)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="steel_certificates.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📈 Export Analytics Report"):
                report = generate_analytics_report(documents)
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name="analytics_report.html",
                    mime="text/html"
                )

def training_data_page():
    """Training data management page"""
    st.title("🎓 Training Data Management")
    
    training_data = st.session_state.db.get_training_data()
    
    if training_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(training_data))
        
        with col2:
            correct_count = sum(1 for item in training_data if item.get('is_correct', False))
            st.metric("Correct Extractions", correct_count)
        
        with col3:
            accuracy = correct_count / len(training_data) if training_data else 0
            st.metric("Overall Accuracy", f"{accuracy:.2%}")
        
        with col4:
            unique_fields = len(set(item.get('field_name') for item in training_data))
            st.metric("Trained Fields", unique_fields)
    
    # Add training data form
    st.subheader("➕ Add Training Data")
    
    with st.form("add_training_data"):
        col1, col2 = st.columns(2)
        
        with col1:
            field_name = st.text_input("Field Name")
            extracted_value = st.text_input("Extracted Value")
        
        with col2:
            correct_value = st.text_input("Correct Value")
        
        context = st.text_area("Context (surrounding text)", height=100)
        
        if st.form_submit_button("Add Training Data"):
            if field_name and extracted_value and correct_value and context:
                success = st.session_state.db.store_training_data(
                    field_name, extracted_value, correct_value, context
                )
                if success:
                    st.success("Training data added successfully!")
                    st.rerun()

def settings_page():
    """Settings and configuration page"""
    st.title("⚙️ Settings")
    
    # Database status
    st.subheader("🔗 PostgreSQL Database Connection")
    
    if st.session_state.db.connection_pool:
        st.success("✅ PostgreSQL connected successfully")
        
        if st.button("Test Database Connection"):
            try:
                conn = st.session_state.db.connection_pool.getconn()
                cursor = conn.cursor()
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                cursor.close()
                st.session_state.db.connection_pool.putconn(conn)
                st.success(f"Database connection test passed! PostgreSQL version: {version[0]}")
            except Exception as e:
                st.error(f"Database connection test failed: {str(e)}")
    else:
        st.error("❌ Database connection failed")
    
    # API Configuration
    st.subheader("🔑 API Configuration")
    
    if st.secrets.get("OPENAI_API_KEY"):
        st.success("✅ OpenAI API key configured")
        
        if st.button("Test OpenAI Connection"):
            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                st.success("OpenAI API connection successful!")
            except Exception as e:
                st.error(f"OpenAI API test failed: {str(e)}")
    else:
        st.warning("⚠️ OpenAI API key not configured")
    
    # PostgreSQL Configuration Info
    st.subheader("🗄️ Database Configuration")
    
    with st.expander("PostgreSQL Configuration Guide"):
        st.markdown("""
        **Required Secrets for PostgreSQL:**
        
        Option 1 - Full Connection String:
        ```toml
        POSTGRES_URL = "postgresql://username:password@host:port/database"
        ```
        
        Option 2 - Individual Parameters:
        ```toml
        POSTGRES_HOST = "your-host"
        POSTGRES_DATABASE = "steel_certificates"
        POSTGRES_USER = "your-username"
        POSTGRES_PASSWORD = "your-password"
        POSTGRES_PORT = 5432
        POSTGRES_SSLMODE = "require"
        ```
        
        **Free PostgreSQL Hosting Options:**
        - 🐘 **Supabase** - Free tier with 500MB storage
        - 🚀 **Railway** - Free tier with PostgreSQL
        - 🌟 **Neon** - Serverless PostgreSQL
        - ☁️ **Aiven** - Free PostgreSQL plan
        
        **PDF Processing Requirements:**
        Add one of these to requirements.txt:
        ```txt
        pymupdf>=1.23.0  # Recommended
        # OR
        PyPDF2>=3.0.0   # Alternative
        # OR  
        pdfplumber>=0.7.0  # Alternative
        ```
        """)
    
    # Data Management
    st.subheader("🗄️ Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Database Stats"):
            docs_count = len(st.session_state.db.get_documents())
            training_count = len(st.session_state.db.get_training_data())
            
            st.info(f"""
            **PostgreSQL Database Statistics:**
            - Documents: {docs_count}
            - Training Records: {training_count}
            """)
    
    with col2:
        if st.button("📤 Export All Data"):
            documents = st.session_state.db.get_documents()
            training_data = st.session_state.db.get_training_data()
            
            # Create ZIP file
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add documents
                doc_json = json.dumps(documents, indent=2, default=str)
                zip_file.writestr("documents.json", doc_json)
                
                # Add training data
                training_json = json.dumps(training_data, indent=2, default=str)
                zip_file.writestr("training_data.json", training_json)
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="Download Full Export",
                data=zip_buffer.getvalue(),
                file_name=f"steel_certs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )

# ================================
# MAIN APP
# ================================

def main():
    """Main application with navigation"""
    initialize_session_state()
    
    # Sidebar navigation
    st.sidebar.title("🔬 Steel Certificate System")
    
    # Navigation buttons
    menu_options = [
        ("🏠 Dashboard", "dashboard"),
        ("📤 Upload Documents", "upload"), 
        ("📊 Analytics", "analytics"),
        ("📚 Document Library", "library"),
        ("🎓 Training Data", "training"),
        ("⚙️ Settings", "settings")
    ]
    
    for label, page_key in menu_options:
        if st.sidebar.button(label, use_container_width=True):
            st.session_state.current_page = page_key
            st.rerun()
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 System Status")
    
    if st.session_state.db and st.session_state.db.connection_pool:
        st.sidebar.success("✅ PostgreSQL Connected")
    else:
        st.sidebar.error("❌ Database Disconnected")
    
    if st.secrets.get("OPENAI_API_KEY"):
        st.sidebar.success("✅ OpenAI Available")
    else:
        st.sidebar.warning("⚠️ OpenAI Not Configured")
    
    # Route to pages
    current_page = st.session_state.current_page
    
    if current_page == "dashboard":
        main_dashboard()
    elif current_page == "upload":
        upload_documents_page()
    elif current_page == "analytics":
        analytics_page()
    elif current_page == "library":
        document_library_page()
    elif current_page == "training":
        training_data_page()
    elif current_page == "settings":
        settings_page()

if __name__ == "__main__":
    main()
