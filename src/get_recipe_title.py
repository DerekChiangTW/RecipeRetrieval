import json

# path of layer1.json
path = '/home/ctlin/Downloads/recipe1M_layers/layer1.json'

title_map = {}
count = 0
file = open(path, "r")
for line in file:
    count += 1
    print count
    id_start = line.find("id\": \"")+6
    id_end = line.find('\"',id_start)
    title_start = line.find("title\": \"") + 9
    title_end = line.find('\"', title_start)
    if id_start <0 or id_end<0 or title_start<0 or title_end<0:
        continue

    id = line[id_start: id_end]
    title = line[title_start:title_end]
    title_map[id] = title

with open('title_map.json', 'w') as outfile:
    json.dump(title_map, outfile)




