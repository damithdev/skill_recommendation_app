import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from collections import Counter
import textdistance

from ml.dataframe import DataFrameSingleton
from models.resume_details import ResumeDetails


def get_recommendations(title, indices, cosine_sim, titles):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:10]
    _indices = [i[0] for i in sim_scores]
    return titles.iloc[_indices]


class Recommendation:
    def __init__(self):
        self.df_instance = DataFrameSingleton()

    def predict(self, resume: ResumeDetails):
        skills_df = self.df_instance.dataframe.copy()
        skills_df = skills_df[["title", "Skills"]]
        skills_df.loc[len(skills_df)] = {"title": "UNK", "Skills": str(resume.skills)}
        skills_df['Skills'] = skills_df.Skills.astype('str')

        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0, stop_words='english')
        tfidf_matrix = tf.fit_transform(skills_df['Skills'])
        cosine_sim = cosine_similarity(tfidf_matrix)

        smd = skills_df.reset_index()
        titles = smd['title']

        indices = pd.Series(smd.index, index=smd['title'])
        recommendations = get_recommendations('UNK', indices, cosine_sim, titles)
        resume.set_job_recommendations(recommendations)
        return resume

    def skills_suggest(self, resume: ResumeDetails):
        skills_df = self.df_instance.dataframe.copy()
        skills = list(set(resume.skills))
        adv = []
        cnt = min(len(resume.job_recommendations), 5)
        indxs = resume.job_recommendations.head(cnt).index
        for i in range(1, cnt):
            sk = skills_df.iloc[indxs]["Skills"][indxs[i]].replace(" '", '').replace("'", '')
            sk = sk.split(",")
            sk.extend(skills)
            final_skills = []
            for i in sk:
                list_items = i.strip('[]').split(', ')
                final_skills.extend(list_items)
            SK = []
            for i in final_skills:
                for j in skills:
                    similarity = textdistance.jaccard.normalized_similarity(i, j)
                    if similarity > 0.7:
                        continue
                    else:
                        SK.append(i)

            frequency_dict = Counter(SK)
            sorted_dict = dict(sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True))
            adv.append(list(sorted_dict.keys())[:10])
        resume.skill_recommendations = adv
        return resume
