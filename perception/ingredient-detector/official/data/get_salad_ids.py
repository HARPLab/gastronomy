import json
import pickle
import os

def get_img_paths(Id, prefix, elem):
    def tr(x):
        path = os.path.join(prefix, x[0], x[1], x[2], x[3], x)
        assert os.path.exists(path)
        return path
    img_paths = []
    for img in elem['images']:
        img_paths.append(tr(img['id']))
    return img_paths

check_ingredients = False
if check_ingredients:
    out_name = 'salad2.txt'
else:
    out_name = 'salad1.txt'

id_to_img_paths = {}
f = open('layer1.json', 'r')
h = open('layer2.json', 'r')
g = open(out_name, 'w')

parsed_json = json.load(f)
parsed_json1 = json.load(h)
id_to_idx = {elem['id']: i for i, elem in enumerate(parsed_json1)}
for i, recipe in enumerate(parsed_json):
    #if i % 100 == 0:
    #    print(i)
    found_salad = False
    if 'salad' in recipe['title'].lower():
        print(recipe['title'])
        found_salad = True
    elif check_ingredients:
        for ingr in recipe['ingredients']:
            if 'salad' in ingr['text'].lower():
                print(recipe['title'])
                print(ingr['text'])
                print()
                found_salad = True
                break
    if found_salad and (recipe['id'] in id_to_idx):
        g.write(str(recipe['id']) + '\n')
        id_to_img_paths[recipe['id']] = get_img_paths(recipe['id'],
                recipe['partition'], parsed_json1[id_to_idx[recipe['id']]])

f.close()
h.close()
g.close()
pickle.dump(id_to_img_paths, open('salad1_id_to_img.pkl', 'wb'))
