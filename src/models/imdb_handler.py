# from torchvision import transforms
import logging
import os
import pandas as pd
from transformers import AutoTokenizer
from model import ElectraClassifier
import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


def my_tokenize(X):
    # Tokenize with electra. Input list of texts
    electra_huggingface = "google/electra-small-discriminator"
    tokenizer = AutoTokenizer.from_pretrained(electra_huggingface)
    tokenizer.padding_side = "left"
    encodings = tokenizer(X, truncation=True, padding=True)

    return encodings


class IMDBelectraClassifier(BaseHandler):
    """
    IMDBelectraClassifier handler class. This handler extends class
    ElectraClassifier from model.py, a default handler.
    This handler takes a IMDB review as .txt
    and returns if review is positive or negative.

    Here method postprocess()
    has been overridden while others are reused from parent class.
    """

    def __init__(self):
        super(IMDBelectraClassifier, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """Loads the model.pt file and initialized the model object.
        Instantiates Tokenizer for preprocessor to use
        Loads labels to name mapping file for post-processing inference response
        """
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        logger.info("this is model_dir:" + model_dir)
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )
        logger.info("this is test line 01:")
        # Read model serialize/pt file
        serialized_file = self.manifest["model"]["serializedFile"]
        logger.info("this is test line 02:")
        model_pt_path = os.path.join(model_dir, serialized_file)
        logger.info("this is test line 03:")
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt or pytorch_model.bin file")

        logger.info(os.path.isfile(model_pt_path))
        # Load model
        self.model = ElectraClassifier()
        self.model.load_state_dict(torch.load("/home/model-server/pytorch_model.bin"))
        logger.info("this is test line 04:")
        self.model.to(self.device)
        self.model.eval()
        logger.debug(
            "Transformer model from path {0} loaded successfully".format(model_dir)
        )

        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        logger.info("Received text: '%s'", text)
        logger.info(text.get("string"))
        review = text.get("string")
        encoding = my_tokenize(review)
        encoded_data = encoding
        return encoded_data, data

    def inference(self, inputs):
        data = inputs[0]
        logger.info(data["input_ids"])
        df_ids = pd.DataFrame(data["input_ids"], columns=["input_ids"])
        df_mask = pd.DataFrame(data["attention_mask"], columns=["attention_mask"])
        tensor_ids = torch.tensor(df_ids["input_ids"])
        tensor_mask = torch.tensor(df_mask["attention_mask"])
        tensor_ids = tensor_ids[None, :]
        tensor_mask = tensor_mask[None, :]
        output = self.model(input_ids=tensor_ids, attention_mask=tensor_mask)

        return output

    def postprocess(self, data):
        """The post process of MNIST converts the predicted output response to a label.

        Args:
            data (list): The predicted output from the Inference
            with probabilities is passed to the post-process function
        Returns:
            list : A list of dictionary with predictons and explanations are returned.
        """
        return data.argmax(1).tolist()
