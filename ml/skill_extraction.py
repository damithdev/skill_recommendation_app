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
        self.sim_threshold = .75
        self.out_threshold = .75
        esco_df = pd.read_csv(config.ESCO_FILE_PATH)
        esco_df['jobbert'] = esco_df['jobbert'].apply(self.conv)
        self.esco_df = esco_df

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

        df_sample = pd.DataFrame({"Job Title": ["UNK"]})
        esco_df = self.esco_df.copy()
        pred_labels, result, oks = self._extract(resume_details.preprocessed_sentences, esco_df)
        df_sample['labels'] = [pred_labels]
        df_sample['ok'] = [oks]

        esco_skills = None
        for index, s in df_sample.iterrows():
            esco_skills = [esco_df.loc[x]['label_cleaned'] for x in s['labels']]

        esco_skills.extend(oks)

        resume_details.skills = esco_skills
        resume_details.sample = df_sample
        return resume_details

    def _extract(self, sentences, esco_df):
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

        idxs, sims = self._compute_similarity_mat(skill_embs, self.emb_label)
        for i in range(len(idxs)):
            if sims[i] > self.sim_threshold:
                pred_labels.append(idxs[i])
                res.append((skill_texts[i], esco_df.iloc[idxs[i]]['label_cleaned'], sims[i]))

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

        esco_embs = [x for x in self.esco_df[emb_type]]
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
