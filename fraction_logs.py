import csv
import numpy as np
import matplotlib.pyplot as plt

# csv_path = "/home/asl/plr/backups/int_nav_0.2/fraction_logs/interaction_log.csv"
# episode_length = 1024
# num_episodes = 50
# fractions = np.zeros((num_episodes, episode_length))

# data = []
# with open(csv_path, mode='r') as logs:
#     reader = csv.reader(logs)
#     for rows in reader:
#         data.append(rows[2])

# for i in range(num_episodes):
#     fractions[i,:] = data[i*1024:(i+1)*1024]

# fig, ax = plt.subplots()

# # We change the fontsize of minor ticks label 
# ax.tick_params(axis='both', which='major', labelsize=20)
# ax.tick_params(axis='both', which='minor', labelsize=20)

# fractions = fractions.mean(axis=0)
# plt.plot(fractions, linewidth=5)
# plt.xlabel("Time step", fontsize=20)
# plt.ylabel("Fraction of interacted objects", fontsize=20)
# plt.title("With Interaction", fontsize=20)
# plt.show()

csv_path_1 = "/home/asl/plr/interaction-mapping/object_logs/bread_104_floor5.csv"
csv_path_2 = "/home/asl/plr/interaction-mapping/object_logs/bread_104_floor5_nav.csv"

episode_length = 1024
num_episodes = 1

fractions_1 = np.zeros((num_episodes, episode_length))
fractions_2 = np.zeros((num_episodes, episode_length))

data_1 = []
with open(csv_path_1, mode='r') as logs:
    reader = csv.reader(logs)
    for rows in reader:
        data_1.append(rows[2])

data_2 = []
with open(csv_path_2, mode='r') as logs:
    reader = csv.reader(logs)
    for rows in reader:
        data_2.append(rows[2])

for i in range(num_episodes):
    fractions_1[i,:] = data_1[i*1024:(i+1)*1024]


for i in range(num_episodes):
    fractions_2[i,:] = data_2[i*1024:(i+1)*1024]

fig, ax = plt.subplots()

# We change the fontsize of minor ticks label 
ax.tick_params(axis='both', which='major', labelsize=25)
ax.tick_params(axis='both', which='minor', labelsize=25)

fractions_1 = fractions_1.mean(axis=0)
ax.plot(fractions_1, linewidth=5)

fractions_2 = fractions_2.mean(axis=0)
ax.plot(fractions_2, linewidth=5)
# ax.legend(["with interaction", "without interaction"], loc='lower right', fontsize=30)
# plt.xlabel("Time step", fontsize=30, fontweight='bold')
# plt.ylabel("Fraction of interacted objects", fontsize=30, fontweight='bold')
# plt.title("Object Coverage", fontsize=30, fontweight='bold')
# plt.show()

plt.show()
