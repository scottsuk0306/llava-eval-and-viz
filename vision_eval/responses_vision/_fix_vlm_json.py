import json
from pathlib import Path

if __name__ == "__main__":
    # Load the JSON file
    file_path = Path("./src/responses_vision/yi_vl_34b.json")
    with file_path.open("r") as file:
        data = json.load(file)

    # Fix the JSON file
    data_dict = {}

    for idx, record in enumerate(data):
        # import pdb; pdb.set_trace()

        uid = (
            record["capability"]
            + "_"
            + record["task"]
            + "_"
            + str(record["instance_idx"])
        )
        data_dict[uid] = {
            "capability": record["capability"],
            "task": record["task"],
            "response": record["yi_vl_34b_response"],
        }

    new_file_path = file_path.parent / "Yi-VL-34B_responses.json"
    # Save the fixed JSON file
    with new_file_path.open("w") as file:
        json.dump(data_dict, file, indent=4)
