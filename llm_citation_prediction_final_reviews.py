#!/usr/bin/env python3
"""
Script to predict yearly citation categories using Large Language Models (LLMs) based on PEER REVIEWS.

This variant uses peer review comments instead of paper full text or abstracts.

USAGE:
1. Set your API keys in the configuration section below
2. Choose your model provider by setting MODEL_PROVIDER to "openai", "gemini", or "qwen"
3. Set the MODEL variable to the specific model you want to use:
   - For OpenAI: "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", etc.
   - For Gemini: "gemini-2.5-pro", "gemini-2.5-flash", etc.
   - For Qwen: "qwen-turbo", "qwen-plus", "qwen-max", etc.
4. Run the script: python llm_citation_prediction_final_reviews.py

Categories (Balanced Distribution):
- Low: Less than 3 citations per year (~30% of papers)
- Medium: 3-10 citations per year (~36% of papers)  
- High: 10+ citations per year (~34% of papers)

REQUIREMENTS:
- pip install openai google-generativeai pandas openpyxl tqdm
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
from google import genai

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "citation_predictions_reviews")

# Model configuration - Change these as needed
MODEL_PROVIDER = "gemini"  # Options: "openai", "gemini", or "qwen"

# Model options:
# OpenAI models: "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"
# Gemini models: "gemini-2.5-pro", "gemini-2.5-flash"
# Qwen models: "qwen-turbo", "qwen-plus", "qwen-max"
MODEL = "gemini-2.5-pro"  # Change this to your preferred model

# Sample size for testing (set to None to use all data)
SAMPLE_SIZE = 50  # Use all papers in the dataset

LOG_FILE = os.path.join(SCRIPT_DIR, f"citation_prediction_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Set up logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Console logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Ensure the output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize API clients
OPENAI_API_KEY = None
GEMINI_API_KEY = None
QWEN_API_KEY = None

if MODEL_PROVIDER == "openai":
    client = OpenAI(api_key=OPENAI_API_KEY)
elif MODEL_PROVIDER == "gemini":
    client = genai.Client(api_key=GEMINI_API_KEY)
elif MODEL_PROVIDER == "qwen":
    client = OpenAI(
        api_key=QWEN_API_KEY,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )

def categorize_citations(citations):
    """Convert yearly citations to categories using balanced thresholds"""
    if citations < 3.0:
        return 'Low'
    elif citations < 10.0:
        return 'Medium'
    else:
        return 'High'

def get_peer_reviews_for_doi(row):
    """Get concatenated peer reviews for a given paper based on published_doi and preprint_category.
    
    Logic:
    1. Use preprint_category to determine folder prefix:
       - "synthetic biology" -> synbio_downloads
       - "bioengineering" -> bioeng_downloads
       - anything else -> arxiv_downloads
    2. Use published_doi with '/' replaced by '_' as the subfolder name
    3. Find files named 'peer_review_1.txt', 'peer_review_2.txt', etc. (up to 4)
    4. Concatenate all peer reviews into a single string
    """
    
    # Get the published DOI and category
    published_doi = row.get('published_doi', '')
    preprint_category = row.get('preprint_category', '').lower().strip()
    
    if not published_doi:
        logging.debug("No published_doi found in row")
        return None
    
    # Determine folder prefix based on preprint_category
    if preprint_category == 'synthetic biology':
        folder_prefix = 'synbio_downloads'
    elif preprint_category == 'bioengineering':
        folder_prefix = 'bioeng_downloads'
    else:
        folder_prefix = 'arxiv_downloads'
    
    # Convert published_doi to folder name format (replace / with _)
    doi_folder = published_doi.replace('/', '_')
    
    # Full path
    folder_path = os.path.join(SCRIPT_DIR, folder_prefix, doi_folder)
    
    logging.debug(f"Category: '{preprint_category}' -> Using folder: {folder_prefix}")
    
    if not os.path.exists(folder_path):
        logging.debug(f"Folder not found: {folder_path}")
        return None
    
    logging.info(f"Looking for peer reviews in: {folder_path}")
    
    # Look for peer_review_1.txt, peer_review_2.txt, etc. (up to 4)
    review_texts = []
    for i in range(1, 5):  # Check for peer_review_1.txt through peer_review_4.txt
        review_filename = f"peer_review_{i}.txt"
        review_path = os.path.join(folder_path, review_filename)
        
        if os.path.exists(review_path):
            try:
                with open(review_path, 'r', encoding='utf-8') as f:
                    review_text = f.read().strip()
                    if review_text and len(review_text) > 50:
                        review_texts.append(f"=== REVIEWER {i} ===\n{review_text}\n")
                        logging.debug(f"Found and loaded {review_filename}")
            except Exception as e:
                logging.warning(f"Failed to read {review_path}: {e}")
    
    if review_texts:
        concatenated_reviews = "\n\n".join(review_texts)
        logging.info(f"Successfully concatenated {len(review_texts)} peer reviews ({len(concatenated_reviews)} characters)")
        return concatenated_reviews
    else:
        logging.debug(f"No peer review files found in {folder_path}")
        return None

def generate_citation_prediction_prompt_from_reviews(review_text):
    """Generate a structured prompt for citation prediction based on peer reviews."""
    
    prompt = f"""You are an expert in academic research citation prediction with deep knowledge across multiple scientific disciplines. Based on the PEER REVIEW COMMENTS below, predict how many citations this paper will receive per year.

**CITATION CATEGORIES:**
- Low: Less than 3 citations per year (about 30% of papers - specialized research with limited scope)
- Medium: 3-10 citations per year (about 36% of papers - solid research with moderate impact)  
- High: 10+ citations per year (about 34% of papers - impactful research with broad appeal)

**EVALUATION FACTORS:**
Analyze the peer review comments to assess these factors:

1. **Novelty and Significance**: Do reviewers highlight novel contributions or significant advances?
2. **Practical Impact**: Do reviewers mention practical applications or problem-solving potential?
3. **Methodology Quality**: What do reviewers say about rigor, reproducibility, and methodology?
4. **Field Relevance**: Do reviews suggest relevance to current hot topics or emerging fields?
5. **Interdisciplinary Appeal**: Do reviewers note broader appeal beyond a narrow specialty?
6. **Technical Innovation**: Do reviews mention new methods, tools, or technologies?
7. **Strengths vs. Weaknesses**: Overall balance of positive vs. critical comments
8. **Reviewer Enthusiasm**: Level of excitement or endorsement from reviewers

**PEER REVIEW COMMENTS TO ANALYZE:**
{review_text}

**INSTRUCTIONS:**
Based on your expert analysis of the peer review comments above, predict the yearly citation category for this paper. 
Consider both the strengths highlighted and concerns raised by reviewers.

Respond with ONLY the category name: Low, Medium, or High

Your response:"""
    
    return prompt

def call_llm_api(prompt, model=MODEL, provider=MODEL_PROVIDER):
    """Call the appropriate LLM API (OpenAI, Gemini, or Qwen) with the given prompt."""
    try:
        system_message = "You are an expert academic researcher and bibliometric analyst with extensive experience in predicting research impact based on peer review feedback. You understand how reviewer comments correlate with future citation patterns across multiple scientific disciplines including biotechnology, synthetic biology, bioengineering, and related fields."
        
        if provider == "openai":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
            
        elif provider == "gemini":
            full_prompt = f"{system_message}\n\n{prompt}"
            import time
            time.sleep(31)  # Rate limiting for Gemini
            response = client.models.generate_content(
                model=model, contents=full_prompt
            )
            return response.text.strip()
            
        elif provider == "qwen":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
    except Exception as e:
        logging.error(f"Error calling {provider} API: {e}")
        return "Low"

def parse_llm_response(response_text):
    """Parse LLM response to extract category."""
    response_text = response_text.upper().strip()
    
    if 'HIGH' in response_text:
        return 'High'
    elif 'MEDIUM' in response_text:
        return 'Medium'
    elif 'LOW' in response_text:
        return 'Low'
    else:
        logging.warning(f"Unclear response: {response_text}, defaulting to Low")
        return 'Low'

def predict_citation_category(review_text, paper_id):
    """Predict citation category based on peer reviews."""
    logging.info(f"Predicting citation for paper {paper_id} using peer reviews")
    
    # Generate prediction prompt
    prompt = generate_citation_prediction_prompt_from_reviews(review_text)
    
    # Get prediction from LLM
    response = call_llm_api(prompt)
    
    # Parse response
    predicted_category = parse_llm_response(response)
    
    logging.info(f"Paper {paper_id}: Predicted '{predicted_category}' (Raw response: '{response}')")
    
    return predicted_category, response

def calculate_metrics(predictions, actual):
    """Calculate comprehensive evaluation metrics."""
    if len(predictions) != len(actual):
        raise ValueError("Prediction and actual lists must have same length")
    
    accuracy = sum(1 for pred, act in zip(predictions, actual) if pred == act) / len(actual)
    
    categories = ['Low', 'Medium', 'High']
    category_metrics = {}
    
    for category in categories:
        tp = sum(1 for pred, act in zip(predictions, actual) if pred == category and act == category)
        fp = sum(1 for pred, act in zip(predictions, actual) if pred == category and act != category)
        fn = sum(1 for pred, act in zip(predictions, actual) if pred != category and act == category)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        category_metrics[category] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(1 for act in actual if act == category),
            'predicted_count': sum(1 for pred in predictions if pred == category)
        }
    
    return {
        'accuracy': accuracy,
        'category_metrics': category_metrics
    }

def main():
    """Main function to run citation prediction analysis using peer reviews."""
    logging.info("Starting LLM-based citation prediction analysis (PEER REVIEWS)")
    logging.info(f"Model Provider: {MODEL_PROVIDER}")
    logging.info(f"Model: {MODEL}")
    logging.info(f"Output folder: {OUTPUT_FOLDER}")
    
    # Load data
    logging.info("Loading dataset...")
    try:
        df_abstracts = pd.read_excel('Supplemental File 1.xlsx', sheet_name=0)
        logging.info(f"Loaded data from Supplemental File 1.xlsx with {len(df_abstracts)} papers")
        df = df_abstracts[df_abstracts["has_peer_review"] == "YES"]
        
    except FileNotFoundError as e:
        logging.error(f"Required file not found: {e}")
        return
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return
    
    # Clean data
    df_clean = df.dropna(subset=['yearly_citations']).copy()
    df_clean['citation_category'] = df_clean['yearly_citations'].apply(categorize_citations)
    
    logging.info(f"Dataset after cleaning: {len(df_clean)} papers")
    
    # Show distribution
    category_counts = df_clean['citation_category'].value_counts()
    logging.info("Citation category distribution:")
    for cat, count in category_counts.items():
        logging.info(f"  {cat:>6}: {count:>3} papers ({count/len(df_clean)*100:>5.1f}%)")
    
    # Sample data if needed
    if SAMPLE_SIZE and SAMPLE_SIZE < len(df_clean):
        df_sample = df_clean.sample(n=SAMPLE_SIZE, random_state=42)
        logging.info(f"Using sample of {len(df_sample)} papers")
    else:
        df_sample = df_clean
        logging.info(f"Using all {len(df_sample)} papers")
    
    # Show sample distribution
    sample_counts = df_sample['citation_category'].value_counts()
    logging.info("Sample distribution:")
    for cat, count in sample_counts.items():
        logging.info(f"  {cat:>6}: {count:>3} papers ({count/len(df_sample)*100:>5.1f}%)")
    
    # Collect peer reviews
    logging.info("Collecting peer reviews for papers...")
    review_contents = []
    valid_indices = []
    skipped_papers = 0
    
    for idx, (_, row) in enumerate(df_sample.iterrows()):
        review_text = get_peer_reviews_for_doi(row)
        
        if review_text and len(review_text.strip()) > 100:
            review_contents.append(review_text)
            valid_indices.append(idx)
        else:
            skipped_papers += 1
            logging.debug(f"Skipping paper {idx+1}: No peer reviews available")
    
    logging.info(f"Found peer reviews for {len(review_contents)} papers")
    logging.info(f"Skipped {skipped_papers} papers without peer reviews")
    
    # Update sample data to only include valid papers
    df_valid = df_sample.iloc[valid_indices].reset_index(drop=True)
    actual_categories = df_valid['citation_category'].tolist()
    actual_citations = df_valid['yearly_citations'].tolist()
    
    # Generate predictions
    logging.info(f"Generating predictions for {len(review_contents)} papers using {MODEL_PROVIDER} {MODEL}...")
    
    predictions = []
    raw_responses = []
    failed_predictions = 0
    
    for i, review_text in enumerate(tqdm(review_contents, 
                                        desc="Predicting citations from reviews", 
                                        unit="paper")):
        try:
            predicted_category, raw_response = predict_citation_category(review_text, i+1)
            predictions.append(predicted_category)
            raw_responses.append(raw_response)
        except Exception as e:
            logging.error(f"Error predicting paper {i+1}: {e}")
            predictions.append("Low")
            raw_responses.append(f"Error: {str(e)}")
            failed_predictions += 1
    
    # Calculate metrics
    logging.info("Calculating evaluation metrics...")
    metrics = calculate_metrics(actual_categories, predictions)
    
    valid_sample_counts = pd.Series(actual_categories).value_counts()
    
    # Results
    logging.info(f"\n{'='*60}")
    logging.info(f"CITATION PREDICTION RESULTS (PEER REVIEWS)")
    logging.info(f"{'='*60}")
    logging.info(f"Model: {MODEL_PROVIDER} {MODEL}")
    logging.info(f"Prediction Mode: PEER REVIEWS")
    logging.info(f"Papers sampled: {len(df_sample)}")
    logging.info(f"Papers processed: {len(actual_categories)}")
    logging.info(f"Papers skipped: {skipped_papers}")
    logging.info(f"Failed predictions: {failed_predictions}")
    logging.info(f"Success rate: {((len(predictions) - failed_predictions) / len(predictions) * 100):.1f}%")
    logging.info(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    logging.info(f"\nPer-category performance:")
    for category in ['Low', 'Medium', 'High']:
        cat_metrics = metrics['category_metrics'][category]
        logging.info(f"  {category:>6}: Precision={cat_metrics['precision']:.3f}, "
                    f"Recall={cat_metrics['recall']:.3f}, F1={cat_metrics['f1']:.3f} "
                    f"(Support: {cat_metrics['support']}, Predicted: {cat_metrics['predicted_count']})")
    
    # Baseline comparison
    baseline_accuracy = valid_sample_counts.max() / len(actual_categories)
    improvement = metrics['accuracy'] - baseline_accuracy
    logging.info(f"\nBaseline accuracy (most frequent class): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    logging.info(f"Model improvement over baseline: {improvement:.4f} ({improvement*100:.2f} percentage points)")
    
    # Prepare detailed results
    detailed_results = []
    for i in range(len(actual_categories)):
        row = {
            'paper_index': df_valid.index[i],
            'actual_category': actual_categories[i],
            'predicted_category': predictions[i],
            'correct_prediction': predictions[i] == actual_categories[i],
            'actual_yearly_citations': actual_citations[i],
            'content_type': 'peer_reviews',
            'content_length': len(review_contents[i]),
            'raw_llm_response': raw_responses[i],
            'review_preview': review_contents[i][:500] + "..." if len(review_contents[i]) > 500 else review_contents[i]
        }
        detailed_results.append(row)
    
    results_df = pd.DataFrame(detailed_results)
    
    # Create summary data
    summary_data = {
        'analysis_date': datetime.now().isoformat(),
        'model_provider': MODEL_PROVIDER,
        'model_used': MODEL,
        'prediction_mode': 'peer_reviews',
        'total_papers': len(df_sample),
        'papers_with_reviews': len(review_contents),
        'failed_predictions': failed_predictions,
        'skipped_papers': skipped_papers,
        'overall_accuracy': metrics['accuracy'],
        'baseline_accuracy': baseline_accuracy,
        'improvement_over_baseline': improvement,
        'category_distribution': sample_counts.to_dict(),
        'category_metrics': metrics['category_metrics']
    }
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Excel file
    excel_filename = os.path.join(OUTPUT_FOLDER, f"citation_predictions_reviews_{MODEL_PROVIDER}_{MODEL}_{timestamp}.xlsx")
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        summary_df = pd.DataFrame([{
            'Metric': 'Overall Accuracy',
            'Value': metrics['accuracy'],
            'Percentage': f"{metrics['accuracy']*100:.2f}%"
        }, {
            'Metric': 'Baseline Accuracy',
            'Value': baseline_accuracy,
            'Percentage': f"{baseline_accuracy*100:.2f}%"
        }, {
            'Metric': 'Improvement',
            'Value': improvement,
            'Percentage': f"{improvement*100:.2f} pp"
        }])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    # JSON file
    json_filename = os.path.join(OUTPUT_FOLDER, f"citation_predictions_reviews_{MODEL_PROVIDER}_{MODEL}_{timestamp}.json")
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    # Confusion matrix
    confusion_matrix = {}
    for actual in ['Low', 'Medium', 'High']:
        confusion_matrix[actual] = {}
        for predicted in ['Low', 'Medium', 'High']:
            count = sum(1 for a, p in zip(actual_categories, predictions) 
                       if a == actual and p == predicted)
            confusion_matrix[actual][predicted] = count
    
    logging.info(f"\nConfusion Matrix:")
    logging.info(f"{'':>10} {'Low':>6} {'Med':>6} {'High':>6}")
    for actual in ['Low', 'Medium', 'High']:
        row = f"{actual:>10}"
        for predicted in ['Low', 'Medium', 'High']:
            row += f" {confusion_matrix[actual][predicted]:>6}"
        logging.info(row)
    
    # Final summary
    logging.info(f"\n{'='*60}")
    logging.info(f"ANALYSIS COMPLETED SUCCESSFULLY")
    logging.info(f"{'='*60}")
    logging.info(f"Results saved to:")
    logging.info(f"  📊 Excel: {excel_filename}")
    logging.info(f"  📄 JSON: {json_filename}")
    logging.info(f"  📝 Log: {LOG_FILE}")
    logging.info(f"\nKey Results:")
    logging.info(f"  🎯 Accuracy: {metrics['accuracy']*100:.2f}%")
    logging.info(f"  📈 Improvement: {improvement*100:.2f} percentage points")
    logging.info(f"  📊 Model: {MODEL_PROVIDER} {MODEL}")
    logging.info(f"  📝 Mode: PEER REVIEWS")

if __name__ == "__main__":
    main()
