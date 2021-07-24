i=0
j=0

l = [[k for k in range(0,5)] for m in range(0,5)]

for i in range(0,5):
    for j in range(0,5):
        x = l[i][j]
        
        print (x," ",end="")
    print ("\n")   

""" for i in range(0,5):
    print (l[i][0],l[i][1],l[i][2],l[i][3],l[i][4],) """