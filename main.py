array = []
for i in range(1, 50):
    for j in range(1, 50):
        for g in range(1, 50):
            for h in range(1, 50):
                a = [i, j, g, h]
                if i ** 3 + j ** 3 == g ** 3 + h ** 3 and len(set(a)) == 4:
                    summ = i ** 3 + j ** 3
                    for k in range(1, 99999999):
                        if summ == k:
                            array.append(k)
    print(set(array))
print()
print(set(array))
