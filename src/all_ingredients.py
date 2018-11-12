from data_loader import *
import json
import re

valid_ingredients = {}
partition = ["train", "val", "test"]
img_path = ""


for part in partition:

    data_path = "/home/ctlin/Desktop/cs646proj/code/data/" + part
    loader = ImagerLoader(img_path, data_path=data_path, partition=part)
    total_recipes = loader.__len__()

    # traverse each recipes
    for i in range(total_recipes-1):
        print "current : ", i
        recipe = loader.__getrecipe__(i)
        ingredients = recipe[0][3]
        # travese each ingredients
        for ingr_idx in range(20):
            ingr = int(ingredients[ingr_idx])
            if ingr != 1:
                valid_ingredients[ingr] = True
            else:
                break



#########SAVE###########

with open('ingred_bool.txt', 'w') as outfile:
    json.dump(valid_ingredients, outfile)

#########LOAD###########
with open('ingred_bool.txt') as json_file:
    ingredients = json.load(json_file)

#########CHECK WITH VOCAB###########
with open('./scripts/vocab.txt') as f:
    content = f.readlines()

with open('all_ingredients.txt', 'w') as outfile:
    for i in range(len(content)):
        line = content[i]
        if str(i+2) in ingredients and re.match("^[a-zA-Z_]*$", line) :
            outfile.write(line)


