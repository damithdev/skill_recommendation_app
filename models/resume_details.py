
class ResumeDetails:
    def __init__(self, file_object, filename):
        self.file_object = file_object
        self.filename = filename
        self.extracted_text = None
        self.preprocessed_sentences = []
        self.skills = []
        self.job_recommendations = None
        self.skill_recommendations = None
        self.sample = None

    def set_preprocessed_sentences(self, sentences):
        self.preprocessed_sentences = sentences

    def set_extracted_text(self, text):
        self.extracted_text = text

    def set_skills(self, skills):
        self.skills = skills

    def set_job_recommendations(self, recommendations):
        self.job_recommendations = recommendations
