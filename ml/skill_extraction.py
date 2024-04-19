import nltk
import pandas as pd
import torch
from transformers import BertModel, AutoTokenizer, pipeline
from threading import Lock

import config
from ml.bert.bert_model import CustomBertModel
from models.resume_details import ResumeDetails


class SkillsExtraction:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SkillsExtraction, cls).__new__(cls)
                cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        self.emb_label = 'jobbert'
        self.sim_threshold = .7
        self.out_threshold = .7
        skills_df = pd.read_csv(config.SKILL_FILE_PATH)
        skills_df['jobbert'] = skills_df['jobbert'].apply(self.conv)
        self.skills_df = skills_df

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Loading the ml and tokenizer (failsafe)
        self.model = BertModel.from_pretrained(config.MODEL_BACKBONE)
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_BACKBONE)

        # Load custom ml state, if available
        try:
            self.model = CustomBertModel(config.MODEL_BACKBONE)
            self.model.load_state_dict(torch.load(config.MODEL_FILE_PATH, map_location=self.device))
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_BACKBONE)
        except FileNotFoundError:
            print("Custom ml state file not found. Using default pre-trained ml.")

        self.model.eval()
        self.model.to(self.device)
        self.token_skill_classifier = pipeline(model="jjzha/jobbert_skill_extraction", aggregation_strategy="first",
                                               device=self.device)
        self.token_knowledge_classifier = pipeline(model="jjzha/jobbert_knowledge_extraction",
                                                   aggregation_strategy="first", device=self.device)

    def reinitialize_model(self):
        # Method to reinitialize the ml
        self._initialize_model()

    # Constructor for initialization, if needed
    def __init__(self):
        pass

    # Sample method to add functionality
    def extract_skills(self, resume_details: ResumeDetails):
        if len(resume_details.preprocessed_sentences) == 0:
            raise ValueError("No preprocessed sentences")

        skills_df = self.skills_df.copy()
        phrases = set()
        skills_sec = False
        iter = 0
        for sent in resume_details.preprocessed_sentences:
            if 'Skills' in sent:
                skills_sec = True
                iter = 0

            iter = iter + 1

            if not skills_sec or iter > 2:
                continue
            relevant_phrases = self._extract_relevant_phrases(sent, list(skills_df['labels']))
            for phrase in relevant_phrases:
                phrases.add(phrase)

        resume_details.preprocessed_sentences.extend(phrases)

        a, b, c = self._extract_with_ner(resume_details.preprocessed_sentences, skills_df)
        ner_skills = [skills_df.loc[x]['labels'] for x in a]
        ner_skills.extend(c)

        sent, skills, sim, idx = self._extract_with_jobbert(resume_details.preprocessed_sentences, skills_df)
        unique_skills = list(set(ner_skills + skills))

        resume_details.skills = unique_skills
        return resume_details

    nltk.download('averaged_perceptron_tagger')

    def _extract_relevant_phrases(self,sentence, skill_keywords):
        # Tokenize the sentence into words and tag their parts of speech
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)

        phrases = []
        current_phrase = []
        for word, tag in pos_tags:
            # Add word to current phrase
            current_phrase.append(word)

            # If the word is a noun, consider it the end of a phrase
            if 'NN' in tag:
                phrase = ' '.join(current_phrase)
                phrases.append(phrase)
                current_phrase = []

        # Check if each phrase contains any skill keywords
        relevant_phrases = [phrase for phrase in phrases if any(skill in phrase for skill in skill_keywords)]

        return relevant_phrases
    def _extract_with_jobbert(self,phrases_list, skills_df, threshold=.9):
        res = []
        phrase_embs = []

        for phrase in phrases_list:
            phrase_embs.append(self._get_embedding(phrase).squeeze())

        idxs, sims = self._compute_similarity_mat(phrase_embs, self.emb_label)
        for i in range(len(idxs)):
            if sims[i] > threshold:
                res.append((phrases_list[i], skills_df.iloc[idxs[i]]['labels'], sims[i], i))

        sorted_res = sorted(res, key=lambda r: r[2], reverse=True)

        sent = []
        skill = []
        sim = []
        idx = []

        for r in sorted_res:
            sent.append(r[0])
            skill.append(r[1])
            sim.append(r[2])
            idx.append(r[3])
            # print('=========================')
            # print(f"sentence: {r[0]}\nESCO skill:{r[1]}\nSimilarity:{r[2]:.4f}")

        return sent, skill, sim, idx
    def _extract_with_ner(self, sentences, skills_df):
        """
        Function that processes outputs from pre-trained, ready to use models
        that detect skills as a token classification task. There are two thresholds,
        out_threshold for filtering model outputs and sim_threshold for filtering
        based on vector similarity with ESCO skills
        """
        #     sentences = get_sentences(job)
        pred_labels = []
        res = []
        skill_embs = []
        skill_texts = []
        oks = []
        for sent in sentences:
            skills, ok = self._ner(sent, self.token_skill_classifier, self.token_knowledge_classifier)
            for entity in skills['entities']:
                text = entity['word']
                if entity['score'] > self.out_threshold:
                    skill_embs.append(self._get_embedding(text).squeeze())
                    skill_texts.append(text)

                if entity['score'] > 0.90:
                    for i in ok:
                        oks.append(i["word"])

        if len(skill_embs) == 0:
            raise Exception("Skills extraction failed!")
        idxs, sims = self._compute_similarity_mat(skill_embs, self.emb_label)
        for i in range(len(idxs)):
            if sims[i] > self.sim_threshold:
                pred_labels.append(idxs[i])
                res.append((skill_texts[i], skills_df.iloc[idxs[i]]['labels'], sims[i]))

        return pred_labels, res, oks

    def _ner(self, text, token_skill_classifier, token_knowledge_classifier):
        output_skills = token_skill_classifier(text)
        for result in output_skills:
            if result.get("entity_group"):
                result["entity"] = "Skill"
                del result["entity_group"]

        output_knowledge = token_knowledge_classifier(text)
        for result in output_knowledge:
            if result.get("entity_group"):
                result["entity"] = "Knowledge"
                del result["entity_group"]

        if len(output_skills) > 0:
            output_skills = self._aggregate_span(output_skills)
        if len(output_knowledge) > 0:
            output_knowledge = self._aggregate_span(output_knowledge)

        skills = []
        skills.extend(output_skills)
        #     skills.extend(output_knowledge)
        #     print(output_knowledge)
        return {"text": text, "entities": skills}, output_knowledge

    def _aggregate_span(self, results):
        new_results = []
        current_result = results[0]

        for result in results[1:]:
            if result["start"] == current_result["end"] + 1:
                current_result["word"] += " " + result["word"]
                current_result["end"] = result["end"]
            else:
                new_results.append(current_result)
                current_result = result

        new_results.append(current_result)

        return new_results

    def _get_embedding(self, x):
        x = self.tokenizer(x, return_tensors='pt')
        x = {k: v.to(self.device) for k, v in x.items()}
        out = self.model(x)

        return out.detach().cpu()

    def _compute_similarity_mat(self, emb_mat, emb_type):

        esco_embs = [x for x in self.skills_df[emb_type]]
        esco_vectors = torch.stack(esco_embs)
        emb_vectors = torch.stack(emb_mat)
        norm_esco_vectors = torch.nn.functional.normalize(esco_vectors, p=2, dim=1)
        norm_emb_vecs = torch.nn.functional.normalize(emb_vectors.T, p=2, dim=0)
        cos_similarities = torch.matmul(norm_esco_vectors, norm_emb_vecs)
        max_similarities, max_indices = torch.max(cos_similarities, dim=0)
        return max_indices.numpy(), max_similarities.numpy()

    def conv(self, ten):
        values_str = ten.split("[")[1].split("]")[0].replace("\n", "").strip().split(",")
        values_float = [float(val) for val in values_str]
        tensor = torch.tensor(values_float)
        return tensor

# Flask integration and usage remains the same
