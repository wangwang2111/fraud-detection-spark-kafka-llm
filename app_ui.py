import streamlit as st
st.set_page_config(page_title="Dialogue Classifier", layout="wide")
from utils.agent_api import DeepSeekClassificationAgent, DeepSeekAnalyzer
from utils.st_functions import styled_badge, load_css
from utils.kafka_utils import get_kafka_consumer, get_kafka_producer
from confluent_kafka import KafkaException
import pandas as pd
from pyspark.sql import SparkSession
import os
import json
from dotenv import load_dotenv
from pathlib import Path

# Apply custom CSS to change the color of st.sidebar.info
load_css("public/main.css")

# Initialize Spark session
spark = SparkSession.builder.appName("StreamlitApp").getOrCreate()

# Load API key from .env
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
if not DEEPSEEK_API_KEY:
    raise ValueError("Missing DEEPSEEK_API_KEY in .env file")

# Cache the agent
@st.cache_resource
def load_agent():
    return DeepSeekClassificationAgent(
        model_path="dialogue_classification_model",
    )

agent = load_agent()

# App Layout
st.title("📞 Customer Dialogue Classifier")
st.markdown("AI-powered tool for identifying potential fraud, product issues, or complaints in dialogues.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("AI Creativity", 0.1, 1.0, 0.7)
    show_confidence = st.checkbox("Show confidence scores", True)
    show_history = st.checkbox("Show historical insights", False)

    st.divider()
    st.header("📂 Upload Historical Data")

    uploaded_file = st.file_uploader(
        "Upload CSV with historical dialogues",
        type=["csv"],
        help="Must contain a 'dialogue' column"
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if "dialogue" not in df.columns:
                st.error("CSV must contain a 'dialogue' column.")
            else:
                spark_df = spark.createDataFrame(df)
                agent.historical_data = spark_df
                st.success(f"Loaded {len(df)} records.")
                st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

# Label mapping
label_mapping = {
    0: "Non-Scam (Safe)",
    1: "Potentially Scam",
}

# Tabbed interface
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Dialogue Analysis", 
    "📊 Batch Prediction (CSV)",
    "📡 Real-time Monitoring"
])

with tab1:
    user_input = st.text_area(
        "Enter a customer service dialogue:",
        height=200,
        placeholder="Paste your dialogue here..."
    )
    
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            try:
                agent.analyzer = DeepSeekAnalyzer(DEEPSEEK_API_KEY)
                result = agent.predict_and_get_label(user_input)

                st.subheader("🔎 Prediction Result")
                cols = st.columns(2)

                cols[0].text("Prediction")
                # Use it in your display
                if int(result['prediction']) == 0:
                  # cols[0].badge("Non-Fraudulent (Safe)", icon=":material/check:", color="green")
                  cols[0].markdown(
                      styled_badge("✅ Non-Fraudulent (Safe)", "#4CAF50"),  # green
                      unsafe_allow_html=True
                  )
                elif int(result['prediction']) == 1:
                  cols[0].markdown(
                      styled_badge("⚠️ Potentially Fraudulent", "#F44336"),  # red
                      unsafe_allow_html=True
                  )
                  # cols[0].markdown(":red-badge[⚠️ Potentially Fraudulent]")
              
                if show_confidence and result.get('confidence') is not None:
                    cols[1].metric("Confidence", f"{result['confidence'] * 100:.0f}%")

                result_analysis = agent.classify_and_explain(user_input)
                try:
                    with st.expander("🧠 AI Explanation", expanded=True):
                        st.write(result_analysis['analysis'])
                except:
                    st.error("Cannot get AI explaination!")

                try:
                    if show_history and result.get('historical_insight'):
                        with st.expander("📚 Historical Context"):
                            st.write(result_analysis['historical_insight'])
                except:
                    st.error("Cannot analyze historical insight!")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

with tab2:
    st.write("Upload a CSV with a `dialogue` column to classify multiple entries.")
    
    if uploaded_file:
        if "dialogue" not in df.columns:
            st.warning("Your file must contain a 'dialogue' column.")
        else:
            if st.button("Predict Labels for Uploaded CSV"):
                with st.spinner("Predicting..."):
                    try:
                        results = []
                        for idx, row in df.iterrows():
                            res = agent.predict_and_get_label(row['dialogue'])
                            results.append({
                                "dialogue": row['dialogue'],
                                "predicted_label": label_mapping.get(res['prediction'], res['prediction']),
                                "confidence": f"{res['confidence']*100:.2f}%" if res.get('confidence') else None
                            })

                        result_df = pd.DataFrame(results)
                        st.success("Batch classification complete.")
                        st.dataframe(result_df, use_container_width=True)

                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="predicted_dialogues.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
    else:
        st.info("Upload a CSV file in the sidebar to enable batch classification.")

with tab3:
    st.header("Real-time Dialogue Monitoring")
    
    if 'kafka_running' not in st.session_state:
        st.session_state.kafka_running = False
        st.session_state.messages = []

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Monitoring") and not st.session_state.kafka_running:
            st.session_state.kafka_running = True
            st.rerun()
    
    with col2:
        if st.button("Stop Monitoring") and st.session_state.kafka_running:
            st.session_state.kafka_running = False
            st.rerun()

    if st.session_state.kafka_running:
        status_placeholder = st.empty()
        results_placeholder = st.empty()
        
        consumer = get_kafka_consumer()
        producer = get_kafka_producer()
        
        try:
            while st.session_state.kafka_running:
                msg = consumer.poll(1.0)
                
                if msg is None:
                    continue
                if msg.error():
                    raise KafkaException(msg.error())
                
                dialogue = json.loads(msg.value().decode('utf-8'))
                
                # Process message
                with st.spinner(f"Analyzing message {msg.key()}..."):
                    result = agent.classify_and_explain(dialogue['text'])
                    
                    # Store result
                    st.session_state.messages.append({
                        'id': msg.key(),
                        'dialogue': dialogue['text'],
                        'prediction': result['prediction'],
                        'confidence': result.get('confidence')
                    })
                    
                    # Send to output topic
                    producer.produce(
                        topic=os.getenv('KAFKA_OUTPUT_TOPIC'),
                        key=msg.key(),
                        value=json.dumps({
                            **result,
                            'original_text': dialogue['text']
                        })
                    )
                    producer.flush()
                
                # Update UI
                with status_placeholder:
                    st.success(f"Processed {len(st.session_state.messages)} messages")
                
                with results_placeholder:
                    for msg in st.session_state.messages[-5:]:
                        label = label_mapping.get(msg['prediction'], 'Unknown')
                        confidence = f"{msg['confidence']*100:.1f}%" if msg['confidence'] else 'N/A'
                        st.markdown(f"""
                        <div class="kafka-message">
                            <span class="prediction-badge {label.lower()}">{label}</span>
                            <span class="confidence">{confidence}</span>
                            <div class="dialogue-preview">{msg['dialogue'][:100]}...</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        except KeyboardInterrupt:
            st.session_state.kafka_running = False
        
        finally:
            consumer.close()