from ml.preprocessor import get_preprocessed_sentences
from service.file_service import FileService
from service.recommendation_service import RecommendationService


class ResumeService:
    def __init__(self):
        self.file_service = FileService()

    def _preprocess(self, resume_details):
        """
        Process the CV by extracting text and possibly more.
        """
        extracted_text = self.file_service.extract_text(file_name=resume_details.filename,
                                                        file_object=resume_details.file_object)
        resume_details.set_extracted_text(extracted_text)
        sentences = get_preprocessed_sentences(resume_details.extracted_text)
        resume_details.set_preprocessed_sentences(sentences)
        print(sentences)

    def get_career_recommendations(self, resume_details):
        """
        Generate job recommendations based on the resume details.
        """
        self._preprocess(resume_details)
        rec_service = RecommendationService()
        rec_service.get_job_recommendation(resume_details)
        return resume_details

    def get_skills_recommendations(self, resume_details):
        """
        Perform skill analysis based on the resume details.
        """

        if resume_details.preprocessed_sentences is None:
            self._preprocess(resume_details)
        rec_service = RecommendationService()
        if resume_details.job_recommendations is None:
            rec_service.get_job_recommendation(resume_details)
        rec_service.get_skills_recommendation(resume_details)
        return resume_details
