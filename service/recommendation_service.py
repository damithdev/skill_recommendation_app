from config import MODEL_FILE_PATH
from ml.recommendation import Recommendation
from ml.skill_extraction import SkillsExtraction
from models.resume_details import ResumeDetails


class RecommendationService:

    def __init__(self):
        self.extractor = SkillsExtraction()
        self.recommendation = Recommendation()

    def _extract_skills_from_career(self, resume_details: ResumeDetails):
        return self.extractor.extract_skills(resume_details)

    def get_job_recommendation(self, resume: ResumeDetails):
        resume = self._extract_skills_from_career(resume)
        result = self.recommendation.predict(resume)
        return result

    def get_skills_recommendation(self, resume: ResumeDetails):
        if len(resume.skills) == 0:
            resume = self._extract_skills_from_career(resume)

        if len(resume.job_recommendations) == 0:
            resume = self.recommendation.predict(resume)

        result = self.recommendation.skills_suggest(resume)
        return result
