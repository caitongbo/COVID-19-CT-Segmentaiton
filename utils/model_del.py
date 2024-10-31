import os
 
def get_model(path, model_name, crossVal):

    finename = path + "/modelBest_"+ model_name +".txt"
    results = []

    with open(finename) as file:
        # print(file.read())

        for item in file:
            item
        results = item.replace(' ','').split(',')

    for result in results:
        if result=='':
            results.remove(result)

    if len(results)==crossVal:
        print(results)
        return results
    else:
        print("error")
        return None


def remove_model(path, model_name,crossVal):

    datanames = os.listdir(path)
    models = []
    for i in datanames:
        if '.pth' in i:
            models.append(i)
    a = len(models)

    finename = path + "/modelBest_"+ model_name +".txt"
    results = []

    with open(finename) as file:
        for item in file:
            item
        results = item.replace(' ','').split(',')

    for result in results:
        if result=='':
            results.remove(result)

    for i in range(crossVal):
        results[i] = 'model_'+ model_name + '_crossVal' + str(i+1) + '_' + results[i] +'.pth'
        models.remove(results[i])

    b = len(results)
    c = len(models)
    
    if a==b+c:
        for model in models:
            print('del: ' + path +'/' +model)
            os.remove(path +'/' +model)
    else:
        print('error')

