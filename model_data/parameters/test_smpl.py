import json

with open('./neutral_smpl_with_cocoplus_reg.txt', 'r') as reader:
    model = json.load(reader)
    
for key in model.keys():
    value = model[key]
    print(key)
    if key in ['posedirs','shapedirs']:
        print('shape:',len(value),len(value[0]),len(value[0][0]))
        continue
    else:
        print('shape:',len(value),len(value[0]))
    if key in ['kintree_table']:
        print('content',value[0])
        print('content',value[1])
    
    else:
        print('first element',value[0])