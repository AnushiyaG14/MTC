import streamlit as st
import requests
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import os
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta

load_dotenv()

# Configuration
FLASK_API_URL = f"http://127.0.0.1:{os.getenv('FLASK_PORT', 5000)}/api"

st.set_page_config(
    page_title="AI-Powered Document Parser & Heat Treatment Analyzer",
    page_icon="🤖",
    layout="wide"
)

def check_backend_connection():
    """Check if Flask backend is running"""
    try:
        response = requests.get(f"{FLASK_API_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else {}
    except:
        return False, {}

def main():
    st.title("🤖 AI-Powered Document Parser & Heat Treatment Data Analyzer")
    st.caption("Enhanced with Machine Learning and Qwen2-VL Vision Model")
    
    # Check backend connection
    is_connected, health_info = check_backend_connection()
    
    if not is_connected:
        st.error("❌ **Backend server is not running!**")
        st.info("""
        **To fix this:**
        1. Install dependencies: `pip install -r requirements.txt`
        2. Activate your virtual environment
        3. Run: `python enhanced_backend.py`
        4. Wait for "Starting Enhanced Flask server with ML capabilities" message
        5. Refresh this page
        """)
        st.stop()
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.success("✅ Backend server is connected")
        with col2:
            if health_info.get('ml_enabled'):
                st.success("🧠 ML Models Active")
            if health_info.get('qwen_model_available'):
                st.success("👁️ Qwen2-VL Available")
    
    # Sidebar navigation
    with st.sidebar:
        selected = option_menu(
            "Navigation",
            ["Upload & Parse", "View Documents", "Analytics", "Search", "ML Training", "Model Status"],
            icons=['cloud-upload', 'file-text', 'bar-chart', 'search', 'cpu', 'gear'],
            menu_icon="robot",
            default_index=0
        )
    
    if selected == "Upload & Parse":
        upload_and_parse()
    elif selected == "View Documents":
        view_documents()
    elif selected == "Analytics":
        show_analytics()
    elif selected == "Search":
        search_documents()
    elif selected == "ML Training":
        ml_training_interface()
    elif selected == "Model Status":
        model_status_interface()

def upload_and_parse():
    st.header("📤 Upload & Parse Documents with AI")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=['pdf', 'xlsx', 'xls', 'png', 'jpg', 'jpeg'],
            help="Upload PDF, Excel, or Image files containing heat treatment data"
        )
    
    with col2:
        st.info("**AI Enhancement**\n- Uses Qwen2-VL for image analysis\n- OpenAI GPT-4o for text parsing\n- ML field classification\n- Automatic training data collection")
        
        enable_training = st.checkbox("Enable ML Training", value=True, 
                                    help="Collect this parsing session for ML model improvement")
    
    if uploaded_file is not None:
        st.info(f"📁 **File:** {uploaded_file.name} ({uploaded_file.size:,} bytes)")
        
        # Show file preview for images
        if uploaded_file.type.startswith('image/'):
            st.image(uploaded_file, caption="Document Preview", use_column_width=True)
        
        if st.button("🚀 Parse Document with AI", type="primary"):
            with st.spinner("Processing document with AI models... This may take a few minutes."):
                try:
                    # Prepare file for API
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {'enable_training': str(enable_training).lower()}
                    
                    # Call Flask API with longer timeout
                    response = requests.post(
                        f"{FLASK_API_URL}/parse", 
                        files=files,
                        data=data,
                        timeout=300  # 5 minutes timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result.get('success'):
                            st.success("✅ Document parsed successfully with AI!")
                            display_parsing_results(result['data'])
                        else:
                            st.error(f"❌ Parsing failed: {result.get('error', 'Unknown error')}")
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        st.text(f"Response: {response.text}")
                        
                except requests.exceptions.Timeout:
                    st.error("⏰ Request timeout - document processing took too long")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Connection error - make sure Flask backend is running")
                    st.info("Run `python enhanced_backend.py` in a separate terminal")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

def display_parsing_results(data):
    """Display comprehensive parsing results"""
    st.subheader("📊 AI Parsing Results")
    
    # Enhanced summary with AI metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Heat Records", data['summary']['totalHeatRecords'])
    with col2:
        st.metric("Validation", data['summary']['validationStatus'])
    with col3:
        extraction_method = data['summary'].get('extractionMethod', 'unknown')
        st.metric("AI Method", extraction_method.upper())
    with col4:
        confidence = data['summary'].get('modelConfidence', 0.0)
        st.metric("Confidence", f"{confidence:.2f}")
    
    # Show extraction method details
    if extraction_method == 'qwen2vl':
        st.success("🎯 **Qwen2-VL Vision Model** - Advanced image analysis used")
    elif extraction_method == 'openai':
        st.info("🧠 **OpenAI GPT-4o** - Advanced text parsing used")
    
    # Show validation errors if any
    if data['summary']['validationErrors']:
        with st.expander("⚠️ Validation Issues", expanded=False):
            for error in data['summary']['validationErrors']:
                st.write(f"- {error}")
    
    # Display comprehensive extracted data
    if data['separateRecords']['success']:
        st.subheader("🔥 Heat Records (Comprehensive Data)")
        
        tabs = st.tabs([f"Heat {record['heatRecord'].get('heat_no', 'Unknown')}" for record in data['separateRecords']['records']])
        
        for i, (tab, record) in enumerate(zip(tabs, data['separateRecords']['records'])):
            with tab:
                display_comprehensive_heat_record(record)
        
        # Enhanced download options
        st.subheader("💾 Download Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # JSON download
            json_data = json.dumps(data['separateRecords']['records'], indent=2)
            st.download_button(
                "📄 Download JSON (Complete)",
                json_data,
                f"heat_records_{data.get('document_id', 'unknown')}.json",
                "application/json"
            )
        
        with col2:
            # CSV download
            if data['csvData']['success']:
                df = pd.DataFrame(data['csvData']['csvData'])
                csv = df.to_csv(index=False)
                st.download_button(
                    "📊 Download CSV (Flattened)",
                    csv,
                    f"heat_records_{data.get('document_id', 'unknown')}.csv",
                    "text/csv"
                )
        
        with col3:
            # Raw data download
            raw_data = json.dumps(data, indent=2)
            st.download_button(
                "🔍 Download Raw AI Data",
                raw_data,
                f"raw_extraction_{data.get('document_id', 'unknown')}.json",
                "application/json"
            )

def display_comprehensive_heat_record(record):
    """Display comprehensive heat record with all extracted fields"""
    heat_data = record.get('heatRecord', {})
    
    # Basic information section
    st.write("### 📋 Basic Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Heat No:** {heat_data.get('heat_no', 'N/A')}")
        st.write(f"**Quantity:** {heat_data.get('qty', 'N/A')}")
    with col2:
        st.write(f"**Client:** {record.get('client', 'N/A')}")
        st.write(f"**Component:** {record.get('component', 'N/A')}")
    with col3:
        st.write(f"**Certificate:** {record.get('certificate_number', 'N/A')}")
        st.write(f"**Sample:** {heat_data.get('sample', 'N/A')}")
    
    # Company and standards information
    if record.get('company_name') or record.get('material_specification'):
        st.write("### 🏢 Company & Standards")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Company:** {record.get('company_name', 'N/A')}")
            st.write(f"**Material Spec:** {record.get('material_specification', 'N/A')}")
        with col2:
            st.write(f"**ISO 9001:** {record.get('iso_9001_approved', 'N/A')}")
            st.write(f"**CE 2014/68/EU:** {record.get('ce_2014_68_eu_approved', 'N/A')}")
    
    # Dimensional and testing information
    testing_conditions = record.get('testing_conditions', {})
    impact_test_info = record.get('impact_test_info', {})
    
    if testing_conditions or impact_test_info:
        st.write("### 🔬 Testing Conditions")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Testing Conditions:**")
            if testing_conditions:
                st.json(testing_conditions)
        with col2:
            st.write("**Impact Test Info:**")
            if impact_test_info:
                st.json(impact_test_info)
    
    # Chemical composition chart
    chem_data = heat_data.get('chemical_composition', {})
    if chem_data and any(chem_data.values()):
        st.write("### ⚗️ Chemical Composition")
        
        # Filter out zero values for better visualization
        filtered_chem = {k: v for k, v in chem_data.items() if v and v != 0}
        
        if filtered_chem:
            fig = px.bar(
                x=list(filtered_chem.keys()),
                y=list(filtered_chem.values()),
                title=f"Chemical Composition - Heat {heat_data.get('heat_no', 'Unknown')}",
                labels={'x': 'Element', 'y': 'Percentage (%)'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No chemical composition data available")
    
    # Mechanical properties
    mech_props = heat_data.get('mechanical_properties', {})
    if mech_props:
        st.write("### 🔧 Mechanical Properties")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("YS (MPa)", f"{mech_props.get('YS_mpa', 'N/A')}")
        with col2:
            st.metric("UTS (MPa)", f"{mech_props.get('UTS_mpa', 'N/A')}")
        with col3:
            st.metric("EL (%)", f"{mech_props.get('EL_percent', 'N/A')}")
        with col4:
            st.metric("ROA (%)", f"{mech_props.get('ROA_percent', 'N/A')}")
        
        col5, col6 = st.columns(2)
        with col5:
            st.metric("Hardness (HBW)", f"{mech_props.get('hardness_hbw', 'N/A')}")
        with col6:
            st.metric("Hardness Avg", f"{mech_props.get('hardness_avg', 'N/A')}")
    
    # Impact test results
    impact_results = heat_data.get('impact_test_results', {})
    if impact_results and any(impact_results.values()):
        st.write("### 💥 Impact Test Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test 1", f"{impact_results.get('no_1', 'N/A')}")
        with col2:
            st.metric("Test 2", f"{impact_results.get('no_2', 'N/A')}")
        with col3:
            st.metric("Test 3", f"{impact_results.get('no_3', 'N/A')}")
        with col4:
            st.metric("Temperature", f"{impact_results.get('temperature_c', 'N/A')}")
    
    # Heat treatment process
    heat_treatment = record.get('heat_treatment', {})
    if heat_treatment and any(heat_treatment.values()):
        st.write("### 🔥 Heat Treatment Process")
        st.json(heat_treatment)

def view_documents():
    st.header("📁 View All Documents")
    
    try:
        response = requests.get(f"{FLASK_API_URL}/documents", timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                documents = result['documents']
                
                if documents:
                    # Enhanced document display with ML metrics
                    df_display = pd.DataFrame([
                        {
                            'ID': doc['_id'],
                            'Client': doc.get('client', 'N/A'),
                            'Component': doc.get('component', 'N/A'),
                            'Certificate': doc.get('certificate_number', 'N/A'),
                            'Heat Records': doc.get('total_heat_records', 0),
                            'Status': doc.get('validation_status', 'N/A'),
                            'AI Method': doc.get('extraction_method', 'N/A'),
                            'Confidence': f"{doc.get('model_confidence', 0):.2f}",
                            'Upload Date': doc.get('upload_date', 'N/A')
                        }
                        for doc in documents
                    ])
                    
                    st.dataframe(df_display, use_container_width=True)
                    
                    # Document details
                    selected_doc = st.selectbox(
                        "Select document to view details:",
                        options=documents,
                        format_func=lambda x: f"{x.get('client', 'Unknown')} - {x.get('certificate_number', 'N/A')}"
                    )
                    
                    if selected_doc:
                        show_document_details(selected_doc['_id'])
                else:
                    st.info("No documents found. Upload some documents first!")
            else:
                st.error(f"Error: {result['error']}")
        else:
            st.error("Failed to fetch documents")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def show_document_details(doc_id):
    """Show detailed view of a document"""
    try:
        # Get heat records
        response = requests.get(f"{FLASK_API_URL}/documents/{doc_id}/heat-records", timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                heat_records = result['heat_records']
                
                st.subheader("🔥 Heat Records Details")
                
                for record in heat_records:
                    with st.expander(f"Heat No: {record['heat_no']} (Qty: {record['qty']})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Chemical Composition:**")
                            st.json(record['chemical_composition'])
                        
                        with col2:
                            st.write("**Mechanical Properties:**")
                            st.json(record['mechanical_properties'])
                        
                        if record.get('impact_test_results'):
                            st.write("**Impact Test Results:**")
                            st.json(record['impact_test_results'])
                        
                        if record.get('hardness_measurements'):
                            st.write("**Hardness Measurements:**")
                            st.json(record['hardness_measurements'])
            else:
                st.error(f"Error: {result['error']}")
        
        # Get raw extraction data
        response = requests.get(f"{FLASK_API_URL}/documents/{doc_id}/raw-data", timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                with st.expander("🔍 Raw AI Extraction Data", expanded=False):
                    st.json(result['raw_data'])
                    
    except Exception as e:
        st.error(f"Error: {str(e)}")

def show_analytics():
    st.header("📈 Analytics Dashboard")
    
    try:
        # Get extraction analytics
        response = requests.get(f"{FLASK_API_URL}/analytics/extraction-accuracy", timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                analytics = result['analytics']
                
                # Overall metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Documents", analytics.get('total_documents', 0))
                with col2:
                    st.metric("Successful Extractions", analytics.get('successful_extractions', 0))
                with col3:
                    success_rate = analytics.get('success_rate', 0)
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                with col4:
                    st.metric("Recent (30 days)", analytics.get('recent_documents_30_days', 0))
                
                # Extraction methods performance
                if analytics.get('extraction_methods'):
                    st.subheader("🤖 AI Method Performance")
                    
                    methods_data = analytics['extraction_methods']
                    df_methods = pd.DataFrame(methods_data)
                    
                    if not df_methods.empty:
                        fig = px.bar(
                            df_methods, 
                            x='_id', 
                            y='successful_extractions',
                            title="Successful Extractions by AI Method",
                            labels={'_id': 'AI Method', 'successful_extractions': 'Successful Extractions'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Average confidence by method
                        if 'avg_confidence' in df_methods.columns:
                            fig2 = px.bar(
                                df_methods,
                                x='_id',
                                y='avg_confidence',
                                title="Average Confidence by AI Method",
                                labels={'_id': 'AI Method', 'avg_confidence': 'Average Confidence'}
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                
                st.success("Analytics data loaded successfully!")
            else:
                st.error("Failed to load analytics")
        else:
            st.error("Analytics service unavailable")
            
    except Exception as e:
        st.error(f"Analytics error: {str(e)}")

def search_documents():
    st.header("🔍 Search Documents")
    
    search_query = st.text_input("Search by client, component, certificate, company, or material specification:")
    
    if search_query:
        try:
            response = requests.get(f"{FLASK_API_URL}/search", params={'q': search_query}, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    documents = result['documents']
                    
                    if documents:
                        st.success(f"Found {len(documents)} documents")
                        
                        for doc in documents:
                            with st.expander(f"{doc.get('client', 'Unknown')} - {doc.get('certificate_number', 'N/A')}"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"**Component:** {doc.get('component', 'N/A')}")
                                    st.write(f"**Heat Records:** {doc.get('total_heat_records', 0)}")
                                with col2:
                                    st.write(f"**Status:** {doc.get('validation_status', 'N/A')}")
                                    st.write(f"**AI Method:** {doc.get('extraction_method', 'N/A')}")
                                with col3:
                                    st.write(f"**Confidence:** {doc.get('model_confidence', 0):.2f}")
                                    st.write(f"**Upload Date:** {doc.get('upload_date', 'N/A')}")
                    else:
                        st.info("No documents found matching your search.")
                else:
                    st.error(f"Search error: {result['error']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

def ml_training_interface():
    st.header("🧠 ML Training Interface")
    st.write("Manage and improve the AI models used for document parsing.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Manual Training")
        if st.button("Train Field Classifier", type="primary"):
            with st.spinner("Training field classifier model..."):
                try:
                    response = requests.post(f"{FLASK_API_URL}/train-model", timeout=60)
                    if response.status_code == 200:
                        result = response.json()
                        if result['success']:
                            st.success("✅ Model training completed successfully!")
                        else:
                            st.error(f"❌ Training failed: {result['error']}")
                    else:
                        st.error("❌ Training request failed")
                except Exception as e:
                    st.error(f"❌ Training error: {str(e)}")
        
        st.info("This will retrain the field classification model using all collected data.")
    
    with col2:
        st.subheader("📊 Training Data Export")
        if st.button("Export Training Data"):
            try:
                response = requests.get(f"{FLASK_API_URL}/export/training-data", timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    if result['success']:
                        training_data = result['training_data']
                        df = pd.DataFrame(training_data)
                        
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Training Data CSV",
                            csv,
                            f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv"
                        )
                        
                        st.success(f"Training data exported: {len(training_data)} samples")
                    else:
                        st.error("Export failed")
            except Exception as e:
                st.error(f"Export error: {str(e)}")

def model_status_interface():
    st.header("⚙️ Model Status & Configuration")
    
    try:
        response = requests.get(f"{FLASK_API_URL}/model-status", timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                status = result['status']
                
                # Training data statistics
                st.subheader("📈 Training Data Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", status.get('total_training_samples', 0))
                with col2:
                    st.metric("Valid Samples", status.get('valid_samples', 0))
                with col3:
                    valid_ratio = (status.get('valid_samples', 0) / max(status.get('total_training_samples', 1), 1)) * 100
                    st.metric("Valid Ratio", f"{valid_ratio:.1f}%")
                
                # Model status
                st.subheader("🤖 Model Status")
                col1, col2 = st.columns(2)
                with col1:
                    if status.get('field_classifier_trained'):
                        st.success("✅ Field Classifier: Trained")
                    else:
                        st.warning("⚠️ Field Classifier: Not Trained")
                    
                    if status.get('vectorizer_trained'):
                        st.success("✅ Text Vectorizer: Trained")
                    else:
                        st.warning("⚠️ Text Vectorizer: Not Trained")
                
                with col2:
                    if status.get('qwen_model_loaded'):
                        st.success("✅ Qwen2-VL Model: Loaded")
                    else:
                        st.warning("⚠️ Qwen2-VL Model: Not Loaded")
                    
                    st.info(f"Confidence Threshold: {status.get('confidence_threshold', 0.8)}")
                
                # Model recommendations
                st.subheader("💡 Recommendations")
                total_samples = status.get('total_training_samples', 0)
                
                if total_samples < 10:
                    st.warning("⚠️ Upload more documents to improve model accuracy (minimum 10 recommended)")
                elif total_samples < 50:
                    st.info("ℹ️ Good progress! Upload more documents for better accuracy (50+ recommended)")
                else:
                    st.success("✅ Excellent! You have sufficient training data for accurate models")
                
                if not status.get('field_classifier_trained') and total_samples >= 10:
                    st.info("💡 Consider training the field classifier in the ML Training tab")
                
            else:
                st.error("Failed to get model status")
        else:
            st.error("Model status service unavailable")
            
    except Exception as e:
        st.error(f"Status error: {str(e)}")

if __name__ == "__main__":
    main()