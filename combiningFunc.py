from ftfy import fix_text

import json

# array with first and second training datasets 
json_files = ["/project/pi_hzamani_umass_edu/asalemi/ERSP/nq/nq_train.json", "/project/pi_hzamani_umass_edu/asalemi/ERSP/train_squad.jsonl"]

combined = []

print("is this working?")

for json_file in json_files:
    with open(json_file, "r") as f:
        # Load each line as a separate JSON object
            if json_file.endswith(".json"):
                combined.extend(json.load(f))
                print(len(combined))
                print("is this working?")
            else:
                 for line in f:
                    if line.strip():
                        combined.append(json.loads(line.strip()))

print(len(combined))

with open("/project/pi_hzamani_umass_edu/asalemi/ERSP/combined.json", "w") as f:
    json.dump(combined, f, indent=4)
    print("is this working?")