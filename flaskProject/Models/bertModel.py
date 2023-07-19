from transformers import DistilBertModel, DistilBertTokenizer, BertTokenizer, BertModel


class BERTModel:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')


bert = BERTModel()
