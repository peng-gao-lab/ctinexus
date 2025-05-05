import os
import json
import copy

data_dir = "output/ET-output"
out_dir = "output/ET-output-preprocessed"
os.makedirs(out_dir, exist_ok=True)


def preprocess():
    # Dictionary to track mention_text to mention_id mapping
    mention_id_map = {}
    current_id = 0

    for inFile in os.listdir(data_dir):
        if inFile.endswith('.json'):
            inFilePath = os.path.join(data_dir, inFile)
            with open(inFilePath, 'r') as f:
                js = json.load(f)

            jsr = copy.deepcopy(js)
            jsr["EA"] = {}
            jsr["EA"]["aligned_triplets"] = js["ET"]["typed_triplets"]

            for triple in jsr["EA"]["aligned_triplets"]:
                for key, entity in triple.items():
                    if key in ["subject", "object"]:
                        mention_text = entity["text"]

                        # Check if mention_text already has an ID
                        if mention_text not in mention_id_map:
                            mention_id_map[mention_text] = current_id
                            current_id += 1

                        # Assign the same mention_id for identical mention_text
                        entity["mention_id"] = mention_id_map[mention_text]
                        entity["mention_text"] = entity.pop("text")
                        entity["mention_class"] = entity.pop("class")

                        # Handle mention_class if it's a dictionary
                        if isinstance(entity["mention_class"], dict):
                            entity["mention_class"] = list(entity["mention_class"].keys())[0]

            jsr["EA"]["mentions_num"] = current_id
            outfile_path = os.path.join(out_dir, inFile)

            with open(outfile_path, 'w') as f:
                json.dump(jsr, f, indent=4)
            print(f"{inFile} is preprocessed")


if __name__ == "__main__":
    preprocess()