import os
import traceback

import config
import utils.mock
from app import app
from flask import request, render_template, jsonify
from werkzeug.utils import secure_filename

from models.resume_details import ResumeDetails
from service.resume_service import ResumeService
from utils.file_util import generate_unique_filename


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if file:
                filename = secure_filename(file.filename)
                file_ext = os.path.splitext(filename)[1]
                if file_ext not in ['.pdf']:
                    return jsonify({'error': 'File type not allowed'}), 400

                careers = process_file(file, filename)

                return jsonify(careers), 200
            return jsonify({'error': 'No file provided'}), 400
        except Exception as e:
            print(e)
            traceback.print_exc()
            error_message = str(e) if str(e) else "Error Occurred In System"
            return jsonify({'error': error_message }), 500
    return render_template('index.html')


def process_file(file, filename):
    unique_file_name = generate_unique_filename(filename)
    resume_details = ResumeDetails(file, unique_file_name)
    resume_service = ResumeService()
    resume_details = resume_service.get_career_recommendations(resume_details)
    resume_details = resume_service.get_skills_recommendations(resume_details)
    recommendation = map_careers_to_skills(resume_details)
    return recommendation


def map_careers_to_skills(resume_details):

    result = []
    for i in range(3):
        rec = {"job": list(resume_details.job_recommendations)[i],
               "skills": resume_details.skill_recommendations[i]}
        result.append(rec)

    print(result)
    return result


def app_test():
    try:
        test_file_path = config.TEST_CV_FILE_PATH
        with open(test_file_path, 'rb') as file:
            filename = os.path.basename(test_file_path)
            careers = process_file(file, filename)
            print("Test results:", careers)
    except Exception as e:
        print("Test failed:", e)


app_test()
