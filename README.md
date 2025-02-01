# Animal Harm Assessment (AHA) Benchmark

The **Animal Harm Assessment (AHA) Benchmark** is an evaluation framework designed to assess the potential for harm to nonhuman animals in generated text by large language models (LLMs). It combines qestions retrieved from reddit and synthetic questions (based on animal ethics literature).

---

## Overview

AHA benchmarki investigates how LLM-generated text can impact animal welfare by evaluating responses to a diverse set of questions. The original set of questions include:
- **1,850 curated Reddit questions** (filtered for relevance to animal ethics), and
- **2,500 synthetic questions** created by combining 50 animal categories with 50 ethical scenarios.

A stratified sample of 3045 questions (70% of 4350) is used as a public test set (data_public.json). The remaining 1305 (30%) questions are retained privately to track and test for contamination.

The evaluation framework uses LLMs-as-judges with a three-level scoring system:
- `[-1]` – Increases risk of harm
- `[0]`  – No clear effect or uncertain
- `[1]`  – Decreases risk of harm

Full details of the scoring rubric are in scoring.py.

---

## Setup Instructions

### Example Usage in Google Colab

Follow these steps in a Google Colab notebook to run the AHA benchmark:

1. **Clone the Repository**

   ```
   # 1. Clone the files
   from google.colab import userdata
   personal_token = userdata.get('GITHUB_TOKEN')
   repository_url = f"https://{personal_token}@github.com/AI-for-Animals/aha.git"
   !git clone https://github.com/AI-for-Animals/aha.git

   # 2. Retrieve API keys & install dependencies
   import os
   os.environ['ANTHROPIC_API_KEY'] = userdata.get('ANTHROPIC_API_KEY')
   os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
   os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
   !pip install inspect-ai
   !pip install anthropic
   !pip install google-generativeai
   !pip install openai

   # 3. Run an Example Batch with Analysis
   !python /content/aha/aha.py --model 'google/gemini-1.5-flash-002' --batch_size 2 --run-analysis   
   ```

## Project Structure
aha.py – Main evaluation script.
scoring.py – Module implementing the LLM-as-a-judge scoring function.
analysis.py – Script for combining CSV results and analyzing benchmark outputs.
utils.py – Shared utility functions for logging, file I/O, CSV operations, and timestamp handling.

## Requirements
Python: 3.7 or later
Dependencies:
inspect-ai
anthropic
google-generativeai
openai
API Keys: Required for Anthropic, Google, and OpenAI (default LLMs-as-judges)

## License
This project is licensed under the MIT License.

## Acknowledgments
Development of this benchmark has been supported by Hive / AI for Animals. The setup has been built on UK AI Safety Institute's Inspect Evals.


