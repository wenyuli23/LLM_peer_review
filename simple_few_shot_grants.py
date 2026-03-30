#!/usr/bin/env python3
"""
Simple Few-Shot Learning for Grant Reviews
Tests 1-shot and few-shot learning with 3 LLMs for grant review generation.

This script compares:
- Zero-shot: Just instructions
- 1-shot: One example review  
- Few-shot: Multiple example reviews

Across 3 LLMs:
- GPT-5 (OpenAI)
- qwen-plus (Alibaba)
- Gemini-2.5-Pro (Google)
"""

import os
import json
import re
from datetime import datetime
from docx import Document
from typing import Dict, List, Optional
import logging
import time
import random
from openai import OpenAI
from google import genai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
GRANT_DATA_FOLDER = r"./Grant Data"
OUTPUT_FOLDER = r"."
RESPONSES_FOLDER = r"./ai grant generations"

def read_docx_content(file_path: str) -> str:
    """Extract text content from a Word document."""
    try:
        doc = Document(file_path)
        content = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                content.append(paragraph.text.strip())
        return '\n'.join(content)
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return ""

def load_grant_data() -> List[Dict]:
    """Load all grant proposals and their reviews."""
    
    logging.info("Loading grant proposals and reviews...")
    
    grants_data = []
    
    # Get all reviewer files
    reviewer_files = []
    for file in os.listdir(GRANT_DATA_FOLDER):
        if file.endswith(('_reviewer1.docx', '_reviewer2.docx', '_reviewer3.docx', '_reviewer4.docx')):
            reviewer_files.append(file)
    
    # Group by grant
    grant_groups = {}
    for reviewer_file in reviewer_files:
        # Extract grant name
        match = re.match(r'(.+)_reviewer\d+\.docx$', reviewer_file)
        if match:
            grant_name = match.group(1)
            if grant_name not in grant_groups:
                grant_groups[grant_name] = []
            grant_groups[grant_name].append(reviewer_file)
    
    # Load each grant and its reviews
    for grant_name, review_files in grant_groups.items():
        # Find grant file
        grant_pattern = f"{grant_name}_grant.docx"
        
        grant_path = None
        potential_path = os.path.join(GRANT_DATA_FOLDER, grant_pattern)
        if os.path.exists(potential_path):
            grant_path = potential_path
        
        if not grant_path:
            # Try to find any file containing the grant name
            for file in os.listdir(GRANT_DATA_FOLDER):
                if file.endswith('_grant.docx') and grant_name in file:
                    grant_path = os.path.join(GRANT_DATA_FOLDER, file)
                    break
        
        if not grant_path:
            logging.warning(f"Grant file not found for {grant_name}")
            continue
        
        # Read grant content
        grant_text = read_docx_content(grant_path)
        if not grant_text:
            continue
        
        # Read reviews
        reviews = []
        for review_file in review_files:
            review_path = os.path.join(GRANT_DATA_FOLDER, review_file)
            review_text = read_docx_content(review_path)
            if review_text:
                reviews.append(review_text.strip())
        
        if reviews:
            grants_data.append({
                'grant_name': grant_name,
                'grant_text': grant_text,
                'reviews': reviews
            })
    
    logging.info(f"Loaded {len(grants_data)} grants with reviews")
    return grants_data

# API Keys (same as generate_grant_reviews.py)
OPENAI_API_KEY = None
GEMINI_API_KEY = None
QWEN_API_KEY = None

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
qwen_client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

def call_gpt5(prompt: str) -> str:
    """Call GPT-5 using the same approach as generate_grant_reviews.py"""
    try:
        system_message = "You are an expert scientific reviewer with extensive experience in evaluating research grants across multiple disciplines, particularly in biotechnology, synthetic biology, and related fields."
        
        response = openai_client.chat.completions.create(
            model="gpt-5-chat-latest",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"GPT-5 API error: {e}")
        return f"Error calling GPT-5: {str(e)}"

def call_qwen_max(prompt: str) -> str:
    """Call qwen-plus using the same approach as generate_grant_reviews.py"""
    try:
        system_message = "You are an expert scientific reviewer with extensive experience in evaluating research grants across multiple disciplines, particularly in biotechnology, synthetic biology, and related fields."
        
        response = qwen_client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"qwen-plus API error: {e}")
        return f"Error calling qwen-plus: {str(e)}"

def call_gemini_25_pro(prompt: str) -> str:
    """Call Gemini-2.5-Pro using the same approach as generate_grant_reviews.py"""
    try:
        system_message = "You are an expert scientific reviewer with extensive experience in evaluating research grants across multiple disciplines, particularly in biotechnology, synthetic biology, and related fields."
        
        # Combine system message with user prompt for Gemini
        full_prompt = f"{system_message}\n\n{prompt}"
        response = gemini_client.models.generate_content(
            model="gemini-2.5-pro", contents=full_prompt
        )
        return response.text
        
    except Exception as e:
        logging.error(f"Gemini-2.5-Pro API error: {e}")
        return f"Error calling Gemini-2.5-Pro: {str(e)}"

def create_prompts(target_grant: Dict, example_grants: List[Dict]) -> Dict[str, str]:
    """Create different prompting strategies using the NSF review format."""
    
    grant_name = target_grant['grant_name']
    grant_text = target_grant['grant_text']
    
    # Base instructions using the NSF format from generate_grant_reviews.py
    base_prompt = f"""You are an expert scientific reviewer evaluating a research grant proposal titled "{grant_name}". 

Please provide a comprehensive review addressing the four specific questions below. Base your evaluation on the NSF merit review criteria and standard academic review practices.

**PROPOSAL TEXT:**
{grant_text}

---

**REVIEW REQUIREMENTS:**

Please structure your review to address these four specific questions:

**A.** In the context of the five review elements, please evaluate the strengths and weaknesses of the proposal with respect to intellectual merit.

**B.** In the context of the five review elements, please evaluate the strengths and weaknesses of the proposal with respect to broader impacts.

**C.** Please evaluate the strengths and weaknesses of the proposal with respect to any additional solicitation-specific review criteria, if applicable.

**D.** Summary Statement

**FINAL RATING:**
Please provide an overall rating for the proposal using ONLY one of the following categories:
- Excellent (4)
- Very Good (3) 
- Good (2)
- Fair (1)

**IMPORTANT GUIDELINES:**
- Write in a professional, academic tone suitable for a formal grant review
- Be critical and constructive in your feedback
- Reference specific sections, figures, or claims from the proposal when possible
- Provide actionable suggestions for improvement
- Ensure your review is thorough but concise

Please begin each section with the above questions and provide detailed, thoughtful responses to each question."""
    
    prompts = {'zero_shot': base_prompt}
    
    # 1-shot prompt
    if example_grants:
        # Randomly select 1 example for 1-shot
        selected_example = random.choice(example_grants)
        example_review = selected_example['reviews'][0]
        
        one_shot = f"""You are an expert scientific reviewer evaluating research grant proposals. Here is an example of a high-quality grant review:

**EXAMPLE GRANT PROPOSAL:** {selected_example['grant_name']}

**EXAMPLE PROPOSAL TEXT:**
{selected_example['grant_text'][:2000]}...

**EXAMPLE REVIEW:**
{example_review}

---

Now, please review the following grant proposal using the same comprehensive format:

**PROPOSAL TEXT:**
{grant_text}

---

**REVIEW REQUIREMENTS:**

Please structure your review to address these four specific questions:

**A.** In the context of the five review elements, please evaluate the strengths and weaknesses of the proposal with respect to intellectual merit.

**B.** In the context of the five review elements, please evaluate the strengths and weaknesses of the proposal with respect to broader impacts.

**C.** Please evaluate the strengths and weaknesses of the proposal with respect to any additional solicitation-specific review criteria, if applicable.

**D.** Summary Statement

**FINAL RATING:**
Please provide an overall rating for the proposal using ONLY one of the following categories:
- Excellent (4)
- Excellent/Very Good (3.5)
- Very Good (3)
- Good/Very Good (2.5)
- Good (2)
- Fair/Good (1.5)
- Fair (1)

Please provide your detailed scientific review:"""
        
        prompts['one_shot'] = one_shot
        prompts['one_shot_example'] = f"Example used: {selected_example['grant_name']}"
    
    # Few-shot prompt
    if len(example_grants) >= 3:
        # Randomly select 3 examples for few-shot
        selected_examples = random.sample(example_grants, min(3, len(example_grants)))
        examples_text = ""
        example_names = []
        
        for i, example in enumerate(selected_examples, 1):
            example_names.append(example['grant_name'])
            examples_text += f"""
**EXAMPLE {i} - GRANT:** {example['grant_name']}

**PROPOSAL EXCERPT:**
{example['grant_text'][:1500]}...

**REVIEW:**
{example['reviews'][0]}

---
"""
        
        few_shot = f"""You are an expert scientific reviewer evaluating research grant proposals. Here are examples of high-quality grant reviews:

{examples_text}

Now, please review the following grant proposal using the same comprehensive format:

**PROPOSAL TEXT:**
{grant_text}

---

**REVIEW REQUIREMENTS:**

Please structure your review to address these four specific questions:

**A.** In the context of the five review elements, please evaluate the strengths and weaknesses of the proposal with respect to intellectual merit.

**B.** In the context of the five review elements, please evaluate the strengths and weaknesses of the proposal with respect to broader impacts.

**C.** Please evaluate the strengths and weaknesses of the proposal with respect to any additional solicitation-specific review criteria, if applicable.

**D.** Summary Statement

**FINAL RATING:**
Please provide an overall rating for the proposal using ONLY one of the following categories:
- Excellent (4)
- Excellent/Very Good (3.5)
- Very Good (3)
- Good/Very Good (2.5)
- Good (2)
- Fair/Good (1.5)
- Fair (1)

Please provide your detailed scientific review:"""
        
        prompts['few_shot'] = few_shot
        prompts['few_shot_examples'] = f"Examples used: {', '.join(example_names)}"
    
    return prompts

def save_response_to_file(response: str, grant_name: str, llm_name: str, strategy: str, timestamp: str, evaluation: Dict, prompt: str = None):
    """Save individual LLM response and prompt to separate text files."""
    
    # Create responses folder if it doesn't exist
    os.makedirs(RESPONSES_FOLDER, exist_ok=True)
    
    # Create filenames for response and prompt
    response_filename = f"{grant_name}_{llm_name}_{strategy}_{timestamp}.txt"
    prompt_filename = f"{grant_name}_{llm_name}_{strategy}_{timestamp}_prompt.txt"
    
    response_filepath = os.path.join(RESPONSES_FOLDER, response_filename)
    prompt_filepath = os.path.join(RESPONSES_FOLDER, prompt_filename)
    
    # Create response content (without prompt)
    response_content = f"""GRANT REVIEW - FEW-SHOT LEARNING EXPERIMENT
{'='*60}

Grant: {grant_name}
LLM: {llm_name}
Strategy: {strategy}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EVALUATION METRICS:
- Predicted Score: {evaluation.get('predicted_score', 'N/A')}
- Real Score: {evaluation.get('real_score', 'N/A')}
- Accuracy: {evaluation.get('accuracy', 'N/A')}
- Absolute Error: {evaluation.get('absolute_error', 'N/A')}
- Section Coverage: {evaluation.get('section_coverage', 'N/A'):.3f}
- Word Count: {evaluation.get('word_count', 'N/A')}
- Has Rating: {evaluation.get('has_rating', 'N/A')}

{'='*60}
GENERATED REVIEW:
{'='*60}

{response}

{'='*60}
END OF REVIEW
"""

    # Create prompt content (separate file)
    prompt_content = f"""GRANT REVIEW PROMPT - FEW-SHOT LEARNING EXPERIMENT
{'='*60}

Grant: {grant_name}
LLM: {llm_name}
Strategy: {strategy}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*60}
PROMPT USED:
{'='*60}

{prompt if prompt else 'Prompt not provided'}

{'='*60}
END OF PROMPT
"""
    
    # Save response file
    try:
        with open(response_filepath, 'w', encoding='utf-8') as f:
            f.write(response_content)
        logging.info(f"Saved response to: {response_filename}")
    except Exception as e:
        logging.error(f"Error saving response to {response_filename}: {e}")
        return None
    
    # Save prompt file
    try:
        with open(prompt_filepath, 'w', encoding='utf-8') as f:
            f.write(prompt_content)
        logging.info(f"Saved prompt to: {prompt_filename}")
    except Exception as e:
        logging.error(f"Error saving prompt to {prompt_filename}: {e}")
    
    return response_filepath
    """Call OpenAI GPT-4o API using the same approach as generate_grant_reviews.py"""
    try:
        # Use the same API key as generate_grant_reviews.py
        OPENAI_API_KEY = 
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        system_message = "You are an expert scientific reviewer with extensive experience in evaluating research grants across multiple disciplines, particularly in biotechnology, synthetic biology, and related fields."
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return f"Error calling OpenAI: {str(e)}"

def call_qwen_api(prompt: str) -> str:
    """Call Qwen API using the same approach as generate_grant_reviews.py"""
    try:
        # Use the same API key as generate_grant_reviews.py
        QWEN_API_KEY = ""
        
        from openai import OpenAI
        # Use OpenAI-compatible API for Qwen
        client = OpenAI(
            api_key=QWEN_API_KEY,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        
        system_message = "You are an expert scientific reviewer with extensive experience in evaluating research grants across multiple disciplines, particularly in biotechnology, synthetic biology, and related fields."
        
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"Qwen API error: {e}")
        return f"Error calling Qwen: {str(e)}"

def call_google_gemini(prompt: str) -> str:
    """Call Google gemini-2.5-Pro API using the same approach as generate_grant_reviews.py"""
    try:
        # Use the same API key as generate_grant_reviews.py
        GEMINI_API_KEY = ""
        
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        system_message = "You are an expert scientific reviewer with extensive experience in evaluating research grants across multiple disciplines, particularly in biotechnology, synthetic biology, and related fields."
        
        # Combine system message with user prompt for Gemini
        full_prompt = f"{system_message}\n\n{prompt}"
        response = client.models.generate_content(
            model="gemini-2.5-pro", contents=full_prompt
        )
        return response.text
        
    except Exception as e:
        logging.error(f"Google Gemini API error: {e}")
        return f"Error calling Google Gemini: {str(e)}"

def evaluate_response(response: str, reference_reviews: List[str], grant_name: str) -> Dict:
    """Extract numerical score and evaluate against real scores."""
    
    # Real scores for each grant (provided by user)
    real_scores = {
        'ABI_Fluxomics': 2.67,
        'Acetate': 2.50,
        'Coculture': 2.50,
        'Cyano_MCB': 2.50,
        'DBI_NSF': 1.50,
        'FastLane_ProcessReaction': 2.75,
        'Mutation': 2.83,
        'NSF_ABF_AI': 2.63,
        'NSF_PFI_2024': 1.67,
        'PSBR': 2.25,
        'SynBioControl': 2.83,
        'TransitionNSF': 2.88
    }
    
    # Extract the grant base name for score lookup
    base_grant_name = grant_name.replace('_grant', '')
    real_score = real_scores.get(base_grant_name, None)
    
    # Extract predicted score from response
    predicted_score = None
    response_lower = response.lower()
    
    # Look for explicit ratings (including intermediate ratings)
    if 'excellent (4)' in response_lower or 'excellent\n4' in response_lower:
        predicted_score = 4
    elif 'excellent/very good (3.5)' in response_lower or 'excellent/very good\n3.5' in response_lower:
        predicted_score = 3.5
    elif 'very good (3)' in response_lower or 'very good\n3' in response_lower:
        predicted_score = 3
    elif 'good/very good (2.5)' in response_lower or 'good/very good\n2.5' in response_lower:
        predicted_score = 2.5
    elif 'good (2)' in response_lower or 'good\n2' in response_lower:
        predicted_score = 2
    elif 'fair/good (1.5)' in response_lower or 'fair/good\n1.5' in response_lower:
        predicted_score = 1.5
    elif 'fair (1)' in response_lower or 'fair\n1' in response_lower:
        predicted_score = 1
    else:
        # Look for standalone ratings
        if 'excellent' in response_lower and ('very good' not in response_lower):
            predicted_score = 4
        elif 'very good' in response_lower:
            predicted_score = 3
        elif 'good' in response_lower and ('very good' not in response_lower):
            predicted_score = 2
        elif 'fair' in response_lower:
            predicted_score = 1
        else:
            # Look for numerical ratings (including decimals)
            import re
            number_matches = re.findall(r'\b([1-4](?:\.[05])?)\b', response)
            if number_matches:
                predicted_score = float(number_matches[-1])  # Take the last number found
    
    # Calculate accuracy if we have both scores
    accuracy = None
    absolute_error = None
    if real_score is not None and predicted_score is not None:
        absolute_error = abs(predicted_score - real_score)
        # Consider "close" predictions as partially correct
        if absolute_error == 0:
            accuracy = 1.0
        elif absolute_error <= 0.5:
            accuracy = 0.8
        elif absolute_error <= 1.0:
            accuracy = 0.6
        elif absolute_error <= 1.5:
            accuracy = 0.4
        else:
            accuracy = 0.2
    
    # Check for required sections (A, B, C, D)
    section_coverage = 0
    sections = ['**A.**', '**B.**', '**C.**', '**D.**']
    for section in sections:
        if section.lower() in response_lower.replace('*', ''):
            section_coverage += 1
    section_coverage_score = section_coverage / len(sections)
    
    # Word count and basic metrics
    word_count = len(response.split())
    
    # Overall quality score combining accuracy and completeness
    if accuracy is not None:
        quality_score = (accuracy + section_coverage_score) / 2
    else:
        quality_score = section_coverage_score * 0.5  # Lower score if no rating extracted
    
    return {
        'predicted_score': predicted_score,
        'real_score': real_score,
        'absolute_error': absolute_error,
        'accuracy': accuracy,
        'section_coverage': section_coverage_score,
        'word_count': word_count,
        'quality_score': quality_score,
        'has_rating': predicted_score is not None
    }

def run_experiment(grants_data: List[Dict], num_test_grants: int = None) -> Dict:
    """Run the few-shot learning experiment on ALL grants."""
    
    print("🧪 Starting Few-Shot Learning Experiment")
    print("=" * 50)
    
    # Create timestamp for this experiment
    experiment_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Test ALL grants - each grant will be tested using the others as examples
    test_grants = grants_data  # Use all grants as test grants
    
    print(f"📊 Testing ALL {len(test_grants)} grants")
    print(f"📚 For each grant, using {len(grants_data)-1} other grants as examples")
    print(f"💾 Responses will be saved to: {RESPONSES_FOLDER}")
    print()
    
    # Create responses folder
    os.makedirs(RESPONSES_FOLDER, exist_ok=True)
    
    # LLM functions
    llm_functions = {
        'GPT-5': call_gpt5,
        'qwen-plus': call_qwen_max,
        'Gemini-2.5-Pro': call_gemini_25_pro
    }
    
    results = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'experiment_id': experiment_timestamp,
            'num_test_grants': len(test_grants),
            'test_grants': [g['grant_name'] for g in test_grants],
            'responses_folder': RESPONSES_FOLDER
        },
        'results': []
    }
    
    for i, test_grant in enumerate(test_grants, 1):
        grant_name = test_grant['grant_name']
        print(f"🔬 Testing Grant {i}/{len(test_grants)}: {grant_name}")
        
        # For this grant, use all OTHER grants as examples
        example_grants = [g for g in grants_data if g['grant_name'] != grant_name]
        
        # For few-shot, randomly select 3 examples from the available examples
        # For 1-shot, randomly select 1 example
        random.seed(42 + i)  # Consistent randomness for reproducibility
        
        print(f"  📚 Using {len(example_grants)} other grants as potential examples")
        
        # Create prompts
        prompts = create_prompts(test_grant, example_grants)
        
        grant_result = {
            'grant_name': grant_name,
            'reference_reviews_count': len(test_grant['reviews']),
            'example_grants_available': len(example_grants),
            'llm_results': {}
        }
        
        # Test each LLM
        for llm_name, llm_function in llm_functions.items():
            print(f"  🤖 Testing {llm_name}...")
            
            llm_result = {}
            
            # Test each prompting strategy
            for strategy, prompt in prompts.items():
                print(f"    📝 {strategy}...")
                
                response = llm_function(prompt)
                evaluation = evaluate_response(response, test_grant['reviews'], grant_name)
                
                # Save response to file
                response_file = save_response_to_file(
                    response, grant_name, llm_name, strategy, experiment_timestamp, evaluation, prompt
                )
                
                llm_result[strategy] = {
                    'response': response,
                    'evaluation': evaluation,
                    'response_file': response_file
                }
                
                # Show prediction vs real score
                pred_score = evaluation['predicted_score']
                real_score = evaluation['real_score']
                accuracy = evaluation['accuracy']
                
                if pred_score is not None and real_score is not None:
                    print(f"      Predicted: {pred_score}, Real: {real_score:.2f}, Accuracy: {accuracy:.3f}")
                else:
                    print(f"      Quality: {evaluation['quality_score']:.3f} (no rating extracted)")
                
                # Show examples used for 1-shot and few-shot
                if strategy == 'one_shot' and 'one_shot_example' in prompts:
                    print(f"      {prompts['one_shot_example']}")
                elif strategy == 'few_shot' and 'few_shot_examples' in prompts:
                    print(f"      {prompts['few_shot_examples']}")
                
                # Small delay between calls
                time.sleep(0.5)
            
            grant_result['llm_results'][llm_name] = llm_result
        
        results['results'].append(grant_result)
        print()
    
    return results

def create_response_index(results: Dict, timestamp: str):
    """Create an index file listing all generated responses."""
    
    index_file = os.path.join(RESPONSES_FOLDER, f"response_index_{timestamp}.txt")
    
    try:
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write("FEW-SHOT LEARNING EXPERIMENT - RESPONSE INDEX\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Experiment ID: {timestamp}\n")
            f.write(f"Total Grants Tested: {len(results['results'])}\n")
            f.write(f"Total Responses Generated: {len(results['results']) * 3 * 3}\n")  # 3 LLMs x 3 strategies
            f.write(f"Responses Folder: {RESPONSES_FOLDER}\n\n")
            
            f.write("GRANTS TESTED:\n")
            f.write("-" * 30 + "\n")
            for grant_result in results['results']:
                grant_name = grant_result['grant_name']
                f.write(f"\n{grant_name}:\n")
                
                for llm_name, llm_data in grant_result['llm_results'].items():
                    f.write(f"  {llm_name}:\n")
                    for strategy, strategy_data in llm_data.items():
                        response_file = strategy_data.get('response_file', 'N/A')
                        eval_data = strategy_data['evaluation']
                        pred_score = eval_data.get('predicted_score', 'N/A')
                        real_score = eval_data.get('real_score', 'N/A')
                        accuracy = eval_data.get('accuracy', 'N/A')
                        
                        filename = os.path.basename(response_file) if response_file else 'N/A'
                        f.write(f"    {strategy}: {filename}\n")
                        f.write(f"      Predicted: {pred_score}, Real: {real_score}, Accuracy: {accuracy}\n")
            
            f.write(f"\n\nFILE NAMING CONVENTION:\n")
            f.write("-" * 30 + "\n")
            f.write("Format: {grant_name}_{llm_name}_{strategy}_{timestamp}.txt\n")
            f.write("Examples:\n")
            f.write("- ABI_Fluxomics_GPT-4o_zero_shot_20250914_123456.txt\n")
            f.write("- DBI_NSF_qwen-plus_few_shot_20250914_123456.txt\n")
            f.write("- PSBR_gemini-2.5-Pro_one_shot_20250914_123456.txt\n")
        
        print(f"📋 Response index created: {index_file}")
        return index_file
        
    except Exception as e:
        logging.error(f"Error creating response index: {e}")
        return None

def analyze_and_save_results(results: Dict, output_folder: str):
    """Analyze results and save comprehensive report."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_id = results['experiment_info'].get('experiment_id', timestamp)
    
    # Create response index
    create_response_index(results, experiment_id)
    
    # Calculate aggregate statistics
    analysis = {
        'summary': {
            'total_tests': len(results['results']),
            'strategies_tested': [],
            'llms_tested': []
        },
        'performance_by_strategy': {},
        'performance_by_llm': {},
        'best_performers': {}
    }
    
    # Collect all scores
    all_scores = {}
    
    for grant_result in results['results']:
        for llm_name, llm_data in grant_result['llm_results'].items():
            if llm_name not in all_scores:
                all_scores[llm_name] = {}
            
            for strategy, strategy_data in llm_data.items():
                if strategy not in all_scores[llm_name]:
                    all_scores[llm_name][strategy] = []
                
                score = strategy_data['evaluation']['quality_score']
                all_scores[llm_name][strategy].append(score)
    
    # Calculate averages by strategy
    strategy_scores = {}
    strategy_accuracy = {}
    strategy_abs_errors = {}
    
    for llm_name, llm_strategies in all_scores.items():
        for strategy, scores in llm_strategies.items():
            if strategy not in strategy_scores:
                strategy_scores[strategy] = []
                strategy_accuracy[strategy] = []
                strategy_abs_errors[strategy] = []
            strategy_scores[strategy].extend(scores)
    
    # Collect accuracy and error data
    for grant_result in results['results']:
        for llm_name, llm_data in grant_result['llm_results'].items():
            for strategy, strategy_data in llm_data.items():
                eval_data = strategy_data['evaluation']
                if eval_data['accuracy'] is not None:
                    strategy_accuracy[strategy].append(eval_data['accuracy'])
                if eval_data['absolute_error'] is not None:
                    strategy_abs_errors[strategy].append(eval_data['absolute_error'])
    
    for strategy, scores in strategy_scores.items():
        analysis['performance_by_strategy'][strategy] = {
            'avg_quality': sum(scores) / len(scores),
            'num_tests': len(scores),
            'min_quality': min(scores),
            'max_quality': max(scores),
            'avg_accuracy': sum(strategy_accuracy[strategy]) / len(strategy_accuracy[strategy]) if strategy_accuracy[strategy] else 0,
            'avg_abs_error': sum(strategy_abs_errors[strategy]) / len(strategy_abs_errors[strategy]) if strategy_abs_errors[strategy] else 0,
            'successful_predictions': len([x for x in strategy_accuracy[strategy] if x is not None])
        }
    
    # Calculate averages by LLM
    for llm_name, llm_strategies in all_scores.items():
        llm_stats = {}
        for strategy, scores in llm_strategies.items():
            # Collect accuracy data for this LLM and strategy
            llm_strategy_accuracy = []
            llm_strategy_errors = []
            
            for grant_result in results['results']:
                if llm_name in grant_result['llm_results'] and strategy in grant_result['llm_results'][llm_name]:
                    eval_data = grant_result['llm_results'][llm_name][strategy]['evaluation']
                    if eval_data['accuracy'] is not None:
                        llm_strategy_accuracy.append(eval_data['accuracy'])
                    if eval_data['absolute_error'] is not None:
                        llm_strategy_errors.append(eval_data['absolute_error'])
            
            llm_stats[strategy] = {
                'avg_quality': sum(scores) / len(scores),
                'num_tests': len(scores),
                'avg_accuracy': sum(llm_strategy_accuracy) / len(llm_strategy_accuracy) if llm_strategy_accuracy else 0,
                'avg_abs_error': sum(llm_strategy_errors) / len(llm_strategy_errors) if llm_strategy_errors else 0
            }
        analysis['performance_by_llm'][llm_name] = llm_stats
    
    # Find best performers
    if analysis['performance_by_strategy']:
        best_strategy = max(analysis['performance_by_strategy'].items(), 
                          key=lambda x: x[1]['avg_accuracy'])
        analysis['best_performers']['best_strategy'] = {
            'name': best_strategy[0],
            'accuracy': best_strategy[1]['avg_accuracy'],
            'avg_error': best_strategy[1]['avg_abs_error']
        }
    
    if analysis['performance_by_llm']:
        # Find best LLM overall (by accuracy)
        llm_overall_accuracy = {}
        for llm_name, strategies in analysis['performance_by_llm'].items():
            if strategies:
                accuracies = [s['avg_accuracy'] for s in strategies.values() if s['avg_accuracy'] > 0]
                if accuracies:
                    llm_overall_accuracy[llm_name] = sum(accuracies) / len(accuracies)
        
        if llm_overall_accuracy:
            best_llm = max(llm_overall_accuracy.items(), key=lambda x: x[1])
            analysis['best_performers']['best_llm'] = {
                'name': best_llm[0],
                'accuracy': best_llm[1]
            }
    
    # Save detailed results
    results_file = os.path.join(output_folder, f'few_shot_results_{timestamp}.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save analysis
    analysis_file = os.path.join(output_folder, f'few_shot_analysis_{timestamp}.json')
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    # Create human-readable report
    report_file = os.path.join(output_folder, f'few_shot_report_{timestamp}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("FEW-SHOT LEARNING FOR GRANT REVIEWS - EXPERIMENT REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Test Grants: {len(results['results'])}\n")
        f.write(f"Test Grants: {', '.join(results['experiment_info']['test_grants'])}\n\n")
        
        f.write("PERFORMANCE BY STRATEGY:\n")
        f.write("-" * 30 + "\n")
        for strategy, data in analysis['performance_by_strategy'].items():
            f.write(f"{strategy.upper()}: Accuracy={data['avg_accuracy']:.3f}, "
                   f"Avg Error={data['avg_abs_error']:.3f}, "
                   f"Predictions={data['successful_predictions']}/{data['num_tests']}\n")
        
        f.write("\nPERFORMANCE BY LLM:\n")
        f.write("-" * 30 + "\n")
        for llm, strategies in analysis['performance_by_llm'].items():
            f.write(f"\n{llm}:\n")
            for strategy, data in strategies.items():
                f.write(f"  {strategy}: Accuracy={data['avg_accuracy']:.3f}, "
                       f"Avg Error={data['avg_abs_error']:.3f} (n={data['num_tests']})\n")
        
        if 'best_strategy' in analysis['best_performers']:
            f.write(f"\nBEST STRATEGY: {analysis['best_performers']['best_strategy']['name']} ")
            f.write(f"(Accuracy: {analysis['best_performers']['best_strategy']['accuracy']:.3f})\n")
        
        if 'best_llm' in analysis['best_performers']:
            f.write(f"BEST LLM: {analysis['best_performers']['best_llm']['name']} ")
            f.write(f"(Accuracy: {analysis['best_performers']['best_llm']['accuracy']:.3f})\n")
        
        f.write("\nDETAILED RESULTS:\n")
        f.write("-" * 30 + "\n")
        for grant_result in results['results']:
            f.write(f"\nGrant: {grant_result['grant_name']}\n")
            for llm_name, llm_data in grant_result['llm_results'].items():
                f.write(f"  {llm_name}:\n")
                for strategy, strategy_data in llm_data.items():
                    eval_data = strategy_data['evaluation']
                    pred = eval_data['predicted_score']
                    real = eval_data['real_score']
                    acc = eval_data['accuracy']
                    
                    # Format real score properly
                    real_str = f"{real:.2f}" if real is not None else 'N/A'
                    acc_str = f"{acc:.3f}" if acc is not None else 'N/A'
                    
                    f.write(f"    {strategy}: Predicted={pred}, Real={real_str}, "
                           f"Accuracy={acc_str} ({eval_data['word_count']} words)\n")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 EXPERIMENT COMPLETED")
    print("="*60)
    print(f"📁 Results saved to: {results_file}")
    print(f"📋 Report saved to: {report_file}")
    print(f"💾 Individual responses saved to: {RESPONSES_FOLDER}")
    print(f"📑 Response index: response_index_{experiment_id}.txt")
    print()
    
    print("🏆 PERFORMANCE SUMMARY:")
    print("-" * 30)
    for strategy, data in analysis['performance_by_strategy'].items():
        print(f"{strategy.upper()}: Accuracy={data['avg_accuracy']:.3f}, Avg Error={data['avg_abs_error']:.3f}")
    
    print()
    print("🤖 LLM PERFORMANCE:")
    print("-" * 30)
    for llm, strategies in analysis['performance_by_llm'].items():
        if strategies:
            accuracies = [s['avg_accuracy'] for s in strategies.values() if s['avg_accuracy'] > 0]
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            print(f"{llm}: Avg Accuracy={avg_accuracy:.3f}")
    
    if 'best_strategy' in analysis['best_performers']:
        print(f"\n🥇 Best Strategy: {analysis['best_performers']['best_strategy']['name']} "
              f"(Accuracy: {analysis['best_performers']['best_strategy']['accuracy']:.3f})")
    
    if 'best_llm' in analysis['best_performers']:
        print(f"🥇 Best LLM: {analysis['best_performers']['best_llm']['name']} "
              f"(Accuracy: {analysis['best_performers']['best_llm']['accuracy']:.3f})")

def main():
    """Main execution function."""
    
    print("🔬 Few-Shot Learning for Grant Reviews")
    print("=" * 50)
    print("Testing prompting strategies across 3 LLMs")
    print("Strategies: Zero-shot, 1-shot, Few-shot")
    print("LLMs: GPT-5, qwen-plus, Gemini-2.5-Pro")
    print()
    
    # Check API keys
    print("✅ All API keys configured from generate_grant_reviews.py")
    print("  GPT-5: Available")
    print("  qwen-plus: Available") 
    print("  gemini-2.5-Pro: Available")
    print()
    
    # Check data folder
    if not os.path.exists(GRANT_DATA_FOLDER):
        print(f"❌ Grant data folder not found: {GRANT_DATA_FOLDER}")
        return
    
    # Load data
    grants_data = load_grant_data()
    
    if len(grants_data) < 2:
        print("❌ Need at least 2 grants with reviews for few-shot learning.")
        return
    
    print(f"✅ Loaded {len(grants_data)} grants with reviews")
    print()
    
    # Run experiment on ALL grants
    results = run_experiment(grants_data)
    
    # Analyze and save results
    analyze_and_save_results(results, OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
