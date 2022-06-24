import argparse
import csv
import logging
import os
import sys



def dataset_output(data_dir,task_name,split):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)


    class InputExample(object):
        """A single training/test example for simple sequence classification."""

        def __init__(self, guid, text_a, text_b=None, label=None):
            """Constructs a InputExample.

            Args:
                guid: Unique id for the example.
                text_a: string. The untokenized text of the first sequence. For single
                sequence tasks, only this sequence must be specified.
                text_b: (Optional) string. The untokenized text of the second sequence.
                Only must be specified for sequence pair tasks.
                label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
            """
            self.guid = guid
            self.text_a = text_a
            self.text_b = text_b
            self.label = label



    class DataProcessor(object):
        """Base class for data converters for sequence classification data sets."""

        def get_train_examples(self, data_dir):
            """Gets a collection of `InputExample`s for the train set."""
            raise NotImplementedError()

        def get_dev_examples(self, data_dir):
            """Gets a collection of `InputExample`s for the dev set."""
            raise NotImplementedError()

        def get_labels(self):
            """Gets the list of labels for this data set."""
            raise NotImplementedError()

        @classmethod#   装饰器  相当于classmethod（_read_tsv）-->当调用_read_tsv（）先执行classmethod所对应的函数再执行_read_tsv（） --》把_read_tsv（）变成类方法
        def _read_tsv(cls, input_file, quotechar=None):
            """Reads a tab separated value file."""
            with open(input_file, "r",encoding='utf-8') as f:
                reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
                lines = []
                for line in reader:
                    if sys.version_info[0] == 2:
                        line = list(str(cell, 'utf-8') for cell in line)
                    lines.append(line)
                return lines

    class StsbProcessor(DataProcessor):
        """Processor for the STS-B data set (GLUE version)."""

        def get_train_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

        def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

        def get_labels(self):
            """See base class."""
            return [None]

        def _create_examples(self, lines, set_type):
            """Creates examples for the training and dev sets."""
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, line[0])
                text_a = line[7]
                text_b = line[8]
                label = line[-1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples

    class MrpcProcessor(DataProcessor):
        """Processor for the MRPC data set (GLUE version)."""

        def get_train_examples(self, data_dir):
            """See base class."""
            logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

        def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

        def get_labels(self):
            """See base class."""
            return ["0", "1"]

        def _create_examples(self, lines, set_type):
            """Creates examples for the training and dev sets."""
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = line[3]
                text_b = line[4]
                label = line[0]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples


    class MnliProcessor(DataProcessor):
        """Processor for the MultiNLI data set (GLUE version)."""

        def get_train_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

        def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
                "dev_matched")

        def get_labels(self):
            """See base class."""
            return ["contradiction", "entailment", "neutral"]

        def _create_examples(self, lines, set_type):
            """Creates examples for the training and dev sets."""
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, line[0])
                text_a = line[8]
                text_b = line[9]
                label = line[-1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples


    class ColaProcessor(DataProcessor):
        """Processor for the Cola data set (GLUE version)."""

        def get_train_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

        def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

        def get_labels(self):
            """See base class."""
            return ["0", "1"]

        def _create_examples(self, lines, set_type):
            """Creates examples for the training and dev sets."""
            examples = []
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                text_a = line[3]
                label = line[1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            return examples


    class SstProcessor(DataProcessor):
        """Processor for the SST-2 data set (GLUE version)."""

        def get_train_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

        def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

        def get_labels(self):
            """See base class."""
            return ["0", "1"]

        def _create_examples(self, lines, set_type):
            """Creates examples for the training and dev sets."""
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label = line[1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            return examples


    class QqpProcessor(DataProcessor):
        """Processor for the QQP data set (GLUE version)."""

        def get_train_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

        def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

        def get_labels(self):
            """See base class."""
            return ["0", "1"]

        def _create_examples(self, lines, set_type):
            """Creates examples for the training and dev sets."""
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                try:
                    text_a = line[3]
                    text_b = line[4]
                    label = line[5]
                except IndexError:
                    continue
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples


    class QnliProcessor(DataProcessor):
        """Processor for the QNLI data set (GLUE version)."""

        def get_train_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

        def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")),
                "dev_matched")

        def get_labels(self):
            """See base class."""
            return ["entailment", "not_entailment"]

        def _create_examples(self, lines, set_type):
            """Creates examples for the training and dev sets."""
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, 1)
                text_a = line[1]
                text_b = line[2]
                label = line[-1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples


    class RteProcessor(DataProcessor):
        """Processor for the RTE data set (GLUE version)."""

        def get_train_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

        def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

        def get_labels(self):
            """See base class."""
            return ["entailment", "not_entailment"]

        def _create_examples(self, lines, set_type):
            """Creates examples for the training and dev sets."""
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = line[1]
                text_b = line[2]
                label = line[-1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples


    class SnliProcessor(DataProcessor):
        """Processor for the SNLI data set (GLUE version)."""

        def get_train_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

        def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

        def get_labels(self):
            """See base class."""
            return ["contradiction", "entailment", "neutral"]

        def _create_examples(self, lines, set_type):
            """Creates examples for the training and dev sets."""
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = line[7]
                text_b = line[8]
                label = line[-1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples


    class BcsProcessor(DataProcessor):
        """Processor for the fake sentence data set (GLUE version)."""

        def get_train_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

        def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

        def get_labels(self):
            """See base class."""
            return ["0", "1"]

        def _create_examples(self, lines, set_type):
            """Creates examples for the training and dev sets."""
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                label = line[1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            return examples

    class WnliProcessor(DataProcessor):
        """Processor for the WNLI data set (GLUE version)."""

        def get_train_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

        def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

        def get_labels(self):
            """See base class."""
            return ["0", "1"]

        def _create_examples(self, lines, set_type):
            """Creates examples for the training and dev sets."""
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, line[0])
                text_a = line[1]
                text_b = line[2]
                label = line[-1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples


    class XnliProcessor(DataProcessor):
        """Processor for the XNLI data set."""

        def __init__(self):
            self.language = "zh"

        def get_train_examples(self, data_dir):
            """See base class."""
            lines = self._read_tsv(
                os.path.join(data_dir, "multinli",
                             "multinli.train.%s.tsv" % self.language))
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "train-%d" % (i)
                text_a = line[0]
                text_b = line[1]
                label = line[2]
                if label == "contradictory":
                    label = "contradiction"
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples

        def get_dev_examples(self, data_dir):
            """See base class."""
            lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
            examples = []
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "dev-%d" % (i)
                language = line[0]
                if language != self.language:
                    continue
                text_a = line[6]
                text_b = line[7]
                label = line[1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            return examples

        def get_labels(self):
            """See base class."""
            return ["contradiction", "entailment", "neutral"]

        def _create_examples(self, lines, set_type):
            """Creates examples for the training and dev sets."""
            examples = []
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                text_a = line[3]
                label = line[1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            return examples

    class DataProcessor_wiki(object):
        """Base class for data converters for sequence classification data sets."""

        def get_train_examples(self, data_dir):
            """Gets a collection of `InputExample`s for the train set."""
            raise NotImplementedError()

        def get_dev_examples(self, data_dir):
            """Gets a collection of `InputExample`s for the dev set."""
            raise NotImplementedError()

        def get_labels(self):
            """Gets the list of labels for this data set."""
            raise NotImplementedError()

        @classmethod#   装饰器  相当于classmethod（_read_tsv）-->当调用_read_tsv（）先执行classmethod所对应的函数再执行_read_tsv（） --》把_read_tsv（）变成类方法
        def _read_tsv(cls, input_file, quotechar=None):
            """Reads a tab separated value file."""
            with open(input_file, "r",encoding='utf-8') as f:
                reader = csv.reader(f,delimiter="\n", quotechar=quotechar)
                lines = []
                for line in reader:
                    if sys.version_info[0] == 2:
                        line = list(str(cell, 'utf-8') for cell in line)
                    lines.append(line)
                return lines






    class WikiProcessor(DataProcessor_wiki):

        """Processor for the SST-2 data set (GLUE version)."""

        def get_train_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "wiki.txt")), "wiki")

        def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

        def get_labels(self):
            """See base class."""
            return None

        def _create_examples(self, lines, set_type):
            """Creates examples for the training and dev sets."""
            examples = []

            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % (set_type, i)
                text_a=line[0]#.extend(line)
                label = None
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            return examples


    def convert_examples_to_data(examples, label_list):
        """Loads a data file into a list of `InputBatch`s."""
        if label_list is not None:
            label_map = {label: i for i, label in enumerate(label_list)}
        data = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = example.text_a
            tokens_b = None
            if example.text_b is not None:
                tokens_b = example.text_b
            if example.label is not None:
                label_id = label_map[example.label]
            else:label_id=None
            data.append(InputExample(guid = example.guid,text_a = tokens_a,text_b = tokens_b,label = label_id))
        return data

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
    processors = {
        "cola": ColaProcessor,
        "sst": SstProcessor,
        "mrpc": MrpcProcessor,
        "stsb": StsbProcessor,#回归任务
        "qqp": QqpProcessor,
        "mnli": MnliProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "xnli": XnliProcessor,
        "snli": SnliProcessor,
        "bcs": BcsProcessor,
        'wnli':WnliProcessor,
        'wiki':WikiProcessor
    }
    num_labels_task = {
        "cola": 2,
        "mnli": 3,
        "rte": 2,
        "mrpc": 2,
        'sst':2,
        'qnli':2,
        'qqp':2,
        'stsb':'',
        'wnli':3
    }
    task_name = task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    label_list = processor.get_labels()
    if split=='train':
        examples = processor.get_train_examples(data_dir)
        data = convert_examples_to_data(examples, label_list)
    else:
        examples = processor.get_dev_examples(data_dir)
        data = convert_examples_to_data(examples, label_list)
    return data

if __name__ == "__main__":
    #data1=dataset_output(data_dir='./QQP/', task_name='qqp',do_train=False)
    data2 = dataset_output(data_dir='./WNLI/', task_name='wnli', do_train=True)
    #print(data1[17].text_a)
    print(data2[17].text_a)
    print()