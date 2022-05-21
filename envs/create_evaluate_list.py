import csv
from random import randint

evaluate_list_path = "/home/asl/plr/interaction-mapping/envs/config/evaluate_list.csv"
episodes_per_scene = 10

list_of_dicts = []
for i in range(1, 6):
    for j in range(episodes_per_scene):
        episode = randint(0, 1234567)
        list_of_dicts.append({"scene": f"FloorPlan{i}", "episode":episode})


keys = list_of_dicts[0].keys()

with open(evaluate_list_path, "w", newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(list_of_dicts)