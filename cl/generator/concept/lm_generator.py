import spacy
import random
import copy
import tensorflow.compat.v1 as tf
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch

class LMGenerator:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-cased')
        self.model.to('cuda')
        self.model.eval()

    def check_availability(self, sentence):
        def check_availability_sentence(x):
            x = x.numpy().decode('utf-8')
            doc = self.nlp(str(x))
            V_concepts = []
            N_concepts = []
            original_tokens = []
            for token in doc:
                original_tokens.append(token.text_with_ws)
                if token.pos_.startswith('V') and token.is_alpha and not token.is_stop:
                    V_concepts.append(token.text_with_ws)
            for noun_chunk in doc.noun_chunks:
                root_noun = noun_chunk[-1]
                if root_noun.pos_ == "NOUN":
                    N_concepts.append(root_noun.text_with_ws)
            if len(N_concepts) >= 2 or len(V_concepts) >= 2:
                if len(set(N_concepts)) == 1 or len(set(V_concepts)) == 1:
                    return False
                else:
                    return True
            else:
                return False

        if type(sentence) == dict:
            result = tf.py_function(check_availability_sentence, [sentence['text']], [tf.bool])[0]
        else:
            result = tf.py_function(check_availability_sentence, [sentence], [tf.bool])[0]

        return result

    def predict_masked_sent(self, text, top_k=5):
        # Tokenize input
        text = "[CLS] %s [SEP]" % text
        tokenized_text = self.tokenizer.tokenize(text)
        masked_index = tokenized_text.index("[MASK]")
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
        # tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

        # Predict all tokens
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            predictions = outputs[0]

        probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
        top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

        pred_idx = top_k_indices[top_k-1]
        predicted_token = self.tokenizer.convert_ids_to_tokens([pred_idx])[0]

        tokenized_text[masked_index] = predicted_token
        return self.tokenizer.convert_tokens_to_string(tokenized_text[1:-1])


    def lm_generate(self, prompt, top_k):
        doc = self.nlp(str(prompt))
        if len(doc) > 190:
            doc = doc[:190]
        concepts = []

        original_tokens = []
        for token in doc:
            original_tokens.append(token.text_with_ws)
            if token.pos_.startswith('V') and token.is_alpha and not token.is_stop:
                concepts.append(token.text_with_ws)

        for noun_chunk in doc.noun_chunks:
            root_noun = noun_chunk[-1]
            if root_noun.pos_ == "NOUN":
                concepts.append(root_noun.text_with_ws)

        random.shuffle(concepts)

        masked_tokens = []

        for tok in original_tokens:
            if tok == concepts[0]:
                masked_tokens.append('[MASK]')
            else:
                masked_tokens.append(tok.strip())

        assert len(masked_tokens) == len(original_tokens)

        result = ' '.join(masked_tokens)

        return self.predict_masked_sent(result, top_k=top_k)
