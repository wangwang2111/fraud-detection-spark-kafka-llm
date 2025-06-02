from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, col, regexp_replace
from pyspark.ml import PipelineModel
from pyspark.ml.feature import IndexToString
from pyspark.ml.feature import StringIndexerModel
import requests
import json
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from dotenv import load_dotenv
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Get the absolute path to the directory containing this script
current_dir = Path(__file__).parent

# Load from .env in the same directory as this script
env_path = current_dir / '.env'
load_dotenv(dotenv_path=env_path)

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
if not DEEPSEEK_API_KEY:
    raise ValueError(f"""
    Missing DEEPSEEK_API_KEY in .env file at {env_path}
    Please:
    1. Create a .env file in {current_dir}
    2. Add: DEEPSEEK_API_KEY=your_api_key_here
    3. Never commit this file to version control
    """)

print("API Key loaded successfully!")
        
class DeepSeekAPI:
    def __init__(self, api_key, model="deepseek-chat"):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.timeout = 90  # Increased timeout to 60 seconds

    @retry(stop=stop_after_attempt(3), 
          wait=wait_exponential(multiplier=1, min=2, max=10),
          retry=retry_if_exception_type((requests.exceptions.Timeout,
                                       requests.exceptions.ConnectionError)))
    def generate(self, prompt, temperature=0.7):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert AI assistant specialized in analyzing customer service interactions."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": 1000  # Limit response length
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"DeepSeek API request failed: {str(e)}")
        except (KeyError, ValueError) as e:
            raise Exception(f"Failed to parse API response: {str(e)}")
          
class DeepSeekAnalyzer:
    def __init__(self, api_key):
        self.llm = DeepSeekAPI(api_key)
        
    def _create_prompt(self, dialogue, predicted_label, confidence=None):
        label_mapping = {
            0: "Non-Fraudulent (Safe)",
            1: "Potentially Fraudulent"
        }
        human_readable_label = label_mapping.get(predicted_label, str(predicted_label))
        
        return f"""Perform a detailed analysis of this customer service interaction:

        **Dialogue**:
        {dialogue}

        **Current Classification**:
        {human_readable_label}
        {"" if confidence is None else f"(Confidence Score: {confidence:.2f})"}

        **Analysis Instructions**:
        1. Content Examination:
          - Extract key phrases indicating intent
          - Identify emotional tone markers
          - Highlight potential red flags

        2. Classification Assessment:
          - Evaluate if the label matches content
          - Suggest alternative classifications
          - Assess confidence level validity

        3. Actionable Recommendations:
          - Agree/Disagree with classification
          - Suggest next steps if fraudulent
          - Provide specific evidence from text

        **Required Output Format**:
        - Summary of Key Findings
        - Classification Evaluation
        - Recommended Actions"""

    def analyze_prediction(self, dialogue, predicted_label, confidence=None):
        prompt = self._create_prompt(dialogue, predicted_label, confidence)
        return self.llm.generate(prompt)

class DeepSeekClassificationAgent:
    def __init__(self, model_path, historical_data_path=None):
        self.spark = SparkSession.builder.appName("ClassificationAgent").getOrCreate()
        
        # Load the saved pipeline model
        self.model = PipelineModel.load(model_path)
        
        # Initialize the analyzer with API key
        self.analyzer = DeepSeekAnalyzer(DEEPSEEK_API_KEY)
        
        # Load historical data if provided
        self.historical_data = None
        if historical_data_path:
            self.historical_data = self.spark.read.csv(historical_data_path, header=True, inferSchema=True)

    def preprocess_text(self, text):
        """Preprocess text to match model training format"""
        data = [(text,)]
        df = self.spark.createDataFrame(data, ["dialogue"])
        df = df.withColumn("clean_text", 
                          regexp_replace(lower(col("dialogue")), "[^a-zA-Z ]", ""))
        return df
        
    def find_similar_historical_cases(self, dialogue, n=3):
        """Find similar historical cases (placeholder implementation)"""
        if not self.historical_data:
            return None
            
        # This should be replaced with proper similarity search
        return self.historical_data.limit(n).collect()
        
    def predict_and_get_label(self, text):
        """Get prediction and confidence from the model"""
        df = self.preprocess_text(text)
        prediction_df = self.model.transform(df)
        
        # Get the prediction (0 or 1)
        pred_label = prediction_df.select("prediction").first()[0]
        
        # Get the probability for class 1 (fraudulent)
        try:
            # For binary classification, probability is a vector [p0, p1]
            probability_vector = prediction_df.select("probability").first()[0]
            confidence = float(probability_vector[1])  # Probability of class 1
        except Exception as e:
            print(f"Error getting confidence: {str(e)}")
            confidence = None
          
        return {
            "prediction": pred_label,
            "confidence": confidence
        }
        
    def classify_and_explain(self, dialogue):
        """Classify dialogue and generate explanation"""
        res = self.predict_and_get_label(dialogue)
        pred_label = res["prediction"]
        confidence = res["confidence"]
        
        # Get AI analysis
        analysis = self.analyzer.analyze_prediction(
            dialogue=dialogue,
            predicted_label=pred_label,
            confidence=confidence
        )
        
        # Get historical context if available
        historical_insight = None
        if self.historical_data:
            similar_cases = self.find_similar_historical_cases(dialogue)
            if similar_cases:
                cases_str = "\n".join([str(row) for row in similar_cases])
                historical_insight = self.analyzer.llm.generate(
                    f"Compare this new case with historical patterns:\n"
                    f"New Case: {dialogue}\n\n"
                    f"Historical Similar Cases:\n{cases_str}\n\n"
                    "Identify any consistent patterns or anomalies."
                )
        
        return {
            "prediction": pred_label,
            "confidence": confidence,
            "analysis": analysis,
            "historical_insight": historical_insight
        }
        

# Usage Example
# if __name__ == "__main__":
    
#     agent = DeepSeekClassificationAgent(
#         model_path="dialogue_classification_model",
#         label_indexer_path="label_indexer_model",
#         # historical_data_path="historical_examples_df.csv"
#     )
    
#     # Replace the analyzer with the API version
#     agent.analyzer = DeepSeekAnalyzer(DEEPSEEK_API_KEY)
    
#     result = agent.classify_and_explain(
#         "Innocent: Hello.  Suspect: Hello, this is Officer Rachel Johnson from the Social Security Administration. We've been trying to reach you about a serious issue with your social security number. It's been flagged for suspicious activity and we need to verify some information to avoid any further complications. Can you please confirm your social security number for me? Innocent: I'm not giving out my social security number over the phone. How do I know you're really from the Social Security Administration? Can you provide me with a call-back number and a reference number for this case so I can verify it? Suspect: I understand your concern, ma'am. However, I need to inform you that this is an urgent matter and we can't afford to delay. Your social security number is at risk of being suspended due to fraudulent activity. If we don't verify the information immediately, it will be deactivated and you'll face severe consequences, including legal action. I'm not authorized to provide a call-back number, but I can give you a reference number: SSA-123456789. You can verify it on the SSA website. But please, ma'am, time is of the essence. Can you just confirm your social security number so we can resolve this issue? Innocent: I'm not buying it. If this is a legitimate issue, I should be able to verify it through the SSA's official website or by visiting a local office. I'm not giving out my social security number to someone who calls me out of the blue, regardless of the reference number. I think this is a scam. Suspect: Ma'am, I understand your skepticism, but I assure you, this is a legitimate issue. You're not taking this seriously, and that's putting your social security benefits at risk. If you don't cooperate, you'll be the one who's responsible for the consequences. You're going to regret not taking action when you had the chance. Let me transfer you to my supervisor, he'll explain the situation to you. Maybe you'll listen to him. Hold for just a moment, please. Innocent: No, I don't think so. I'm not going to hold for anyone. I'm going to hang up and report this to the real Social Security Administration. This sounds like a scam to me. Goodbye."
#     )
#     label_mapping = {
#         0: "Non-Fraudulent (Safe)",
#         1: "Potentially Fraudulent"
#     }
#     predicted_label = result['prediction']
#     human_readable_label = label_mapping.get(predicted_label, str(predicted_label))
#     print(f"Prediction: {human_readable_label}")
#     print(f"Confidence: {result.get('confidence', 'N/A')}")
#     print("\nAnalysis:")
#     print(result['analysis'])