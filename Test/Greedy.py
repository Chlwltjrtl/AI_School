'''
# GreedyAlgorithm

Yang-Jae-Innovation-hub 
  Ai- hands-on worker -Test
'''


import sys


def find_min_room(num_class, classes):
    # 구현해주세요!
    d = []
    for i in classes:
        a, b, c = i[0], i[1], i[2]
        d.append((a, b))
        d.append((a, c))
    print(d)
    d.sort(key=lambda x: x[1]) # lambda 인자 : 표현식 key = x:x[1] 1에관한 인자로?
    print(d)

    current = [False for _ in range(100010)] # 100010 짜리 False list
    t = 0
    min_room = 0
    for i in range(len(d)):
        x = d[i][0]
        if (not current[x]):
            current[x] = True
            t += 1
        else:
            current[x] = False
            t -= 1
        if (i != len(d) - 1):
            if (d[i][1] < d[i + 1][1]): # a4 가지고 계산해보는게 어떨까
                print(min_room,t)
                min_room = max(min_room, t)
    return min_room


def read_inputs():
    num_class = int(input())
    classes = []
    for i in range(num_class):
        line = [int(x) for x in input().split()]
        class_no = line[0]
        start = line[1]
        end = line[2]
        classes.append((class_no, start, end))

    return num_class, classes  # num_class=강의실번호 , classes=리스트 넣어져있는거


def main():
    num_class, classes = read_inputs()
    ans = find_min_room(num_class, classes)
    print(ans)


if __name__ == "__main__":
    main()
