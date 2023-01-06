import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{tjong-kim-sang-2002-introduction,
    title = "Introduction to the {C}o{NLL}-2002 Shared Task: Language-Independent Named Entity Recognition",
    author = "Tjong Kim Sang, Erik F.",
    booktitle = "{COLING}-02: The 6th Conference on Natural Language Learning 2002 ({C}o{NLL}-2002)",
    year = "2002",
    url = "https://www.aclweb.org/anthology/W02-2024",
}
"""

_DESCRIPTION = """\
Named entities are phrases that contain the names of persons, organizations, locations, times and quantities.
Example:
[PER Wolff] , currently a journalist in [LOC Argentina] , played with [PER Del Bosque] in the final years of the seventies in [ORG Real Madrid] .
The shared task of CoNLL-2002 concerns language-independent named entity recognition.
We will concentrate on four types of named entities: persons, locations, organizations and names of miscellaneous entities that do not belong to the previous three groups.
The participants of the shared task will be offered training and test data for at least two languages.
They will use the data for developing a named-entity recognition system that includes a machine learning component.
Information sources other than the training data may be used in this shared task.
We are especially interested in methods that can use additional unannotated data for improving their performance (for example co-training).
The train/validation/test sets are available in Spanish and Dutch.
For more details see https://www.clips.uantwerpen.be/semeval2016/ner/ and https://www.aclweb.org/anthology/W02-2024/
"""

_URL = "https://raw.githubusercontent.com/YaxinCui/Semeval_2020_task9_data/main/Spanglish/"

TRAINING_FILE_Dict = {
    'Spanglish': "Spanglish_train.conll",

}

TEST_FILE_Dict = {
    'Spanglish': "Spanglish_dev.conll",
}

class Semeval2016Config(datasets.BuilderConfig):
    """BuilderConfig for Semeval2016"""

    def __init__(self, **kwargs):
        """BuilderConfig forSemeval2016.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Semeval2016Config, self).__init__(**kwargs)


class Semeval2016(datasets.GeneratorBasedBuilder):
    """Semeval2016 dataset."""

    BUILDER_CONFIGS = [
        Semeval2016Config(name="Spanglish", version=datasets.Version("1.0.0"), description="Semeval2016 Spanish dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "meta": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    #                     "langs": datasets.Sequence(datasets.features.ClassLabel(names=["lang1","lang2","ambiguous","other","ne","unk","mixed","fw","8","9","10","11",]                         )                     ), 
                    "label": datasets.features.ClassLabel(
                            names=[
                                "positive",
                                "neutral",
                                "negative",
                            ]
                        ),
                }
            ),
            supervised_keys=None,
            homepage="/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        
        if self.config.name=="Spanglish":
            urls_to_download = {
                "train": f"{_URL}{TRAINING_FILE_Dict[self.config.name]}",
                "test": f"{_URL}{TEST_FILE_Dict[self.config.name]}",
            }
        
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        prev_pos = '$$$'
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            meta = None
            tokens = []
            langs = []
            label = None
            for line in f:
                if len(tokens) and (line == "" or line == "\n"):
                    yield guid, {
                        "id": str(guid),
                        "meta": str(meta),
                        "tokens": tokens,
                        "label": label,
                    }
                    guid += 1
                    tokens = []
                    langs = []
                    labels = []
                else:
                    line = line.strip()
                    # semeval2016 tokens are space separated
                    splits = [s.rstrip() for s in line.split("	")]
                    if len(tokens)==0 and line.startswith("meta	"):
                        meta = splits[1]
                        label = splits[2]
                    else:
                        tokens.append(splits[0])
                        langs.append(splits[1])
            # last example
            
            yield guid, {
                "id": str(guid),
                "meta": str(meta),
                "tokens": tokens,
                "label": label,
            }
