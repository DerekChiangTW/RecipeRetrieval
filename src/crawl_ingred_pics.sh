#for ingr in $(cat all_ingredients.txt); do mkdir -p $ingr; done
for ingr in $(cat all_ingredients.txt); do googleimagesdownload -f "png" -s "medium" -t "photo" --keywords $ingr --limit 3; done