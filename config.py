import os
# Configuration settings for the Flask application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the trained ml file
MODEL_BACKBONE = 'jjzha/jobbert-base-cased'
MODEL_FILE_PATH = os.path.join(BASE_DIR, 'ml/files/bert_model_state.pth')
SKILL_FILE_PATH = os.path.join(BASE_DIR, 'ml/files/input/skills_processed.csv')
JOBS_FILE_PATH = os.path.join(BASE_DIR, 'ml/files/input/job_skills_processed.csv')
TEST_CV_FILE_PATH = os.path.join(BASE_DIR, 'ml/files/test/test.pdf')

