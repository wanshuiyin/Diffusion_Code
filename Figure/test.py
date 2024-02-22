list_test = [1,2,3]
# test = input("请输入测试字符串：")
test ='()[{}]'
temp = []


# for i in test: 字符串可以直接看作一个序列并且看它其中的元素
for i in test:
    if i == "(" or "{" or  "[":
        temp.append(i)
    
    else:
        if len(temp) == 0:
            print("非法T_T")
            break

        if i == ')':
            d = '('
        elif i == ']':
            d = '['
        elif i == '}':
            d = '{'

        if d != temp.pop():
            print("非法T_T")
            break
    


if len(temp) == 0:
    print("合法^o^")
else:
    print("非法T_T")