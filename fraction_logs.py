import csv
import numpy as np
import matplotlib.pyplot as plt

csv_path_1 = "/home/asl/plr/backups/int_nav_0.2/fraction_logs/interaction_log.csv"
csv_path_2 = "/home/asl/plr/backups/nav_0.4/fraction_logs/interaction_log.csv"
# csv_path_3 = "/home/asl/plr/backups/int/fraction_logs/interaction_log.csv"

episode_length = 1024
num_episodes = 50

fractions_1 = np.zeros((num_episodes, episode_length))
fractions_2 = np.zeros((num_episodes, episode_length))
# fractions_3 = np.zeros((num_episodes, 1000))

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

# data_3 = []
# with open(csv_path_3, mode='r') as logs:
#     reader = csv.reader(logs)
#     for rows in reader:
#         data_3.append(rows[2])


for i in range(num_episodes):
    fractions_1[i,:] = data_1[i*episode_length:(i+1)*episode_length]


for i in range(num_episodes):
    fractions_2[i,:] = data_2[i*episode_length:(i+1)*episode_length]

# for i in range(num_episodes):
#     fractions_3[i,:] = data_3[i*1000:(i+1)*1000]

fig, ax = plt.subplots()

# We change the fontsize of minor ticks label 
ax.tick_params(axis='both', which='major', labelsize=25)
ax.tick_params(axis='both', which='minor', labelsize=25)

fractions_1 = fractions_1.mean(axis=0)
ax.plot(fractions_1, linewidth=5)

fractions_2 = fractions_2.mean(axis=0)
ax.plot(fractions_2, linewidth=5)

# fractions_3 = fractions_3.mean(axis=0)
# ax.plot(fractions_3, linewidth=5)

# ax.legend([r'$\alpha$=1, $\beta$=0.2 (int+nav)', r'$\alpha$=0, $\beta$=0.2 (nav)', r'$\alpha$=1, $\beta$=0 (int)'], loc='upper left', fontsize=30)
ax.legend([r'$\alpha$=1, $\beta$=0.2 (int+nav)', r'$\alpha$=0, $\beta$=0.2 (nav)'], loc='upper left', fontsize=30)
plt.xlabel("Time step", fontsize=40, fontweight='bold')
plt.ylabel("Fraction of interacted objects", fontsize=40, fontweight='bold')
plt.title("Object Coverage", fontsize=40, fontweight='bold')
plt.show()

# plt.show()
