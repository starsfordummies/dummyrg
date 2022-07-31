li = [1,2,3]

while li[-1] < 10:
    li2 = [len(li),len(li)+1]
    li.extend(li2)
    print(li)

