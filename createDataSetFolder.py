import os
import shutil
n = 0

def f(directory, n):
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            n += 1
            shutil.copyfile(f, r'C:\Users\Влад\Desktop\sp_test\ecommerce products\all\f' + str(n) + '.jpg')
    return n        

d1 = r'C:\Users\Влад\Desktop\sp_test\ecommerce products\jeans'
d2 = r'C:\Users\Влад\Desktop\sp_test\ecommerce products\sofa'
d3 = r'C:\Users\Влад\Desktop\sp_test\ecommerce products\tshirt'
d4 = r'C:\Users\Влад\Desktop\sp_test\ecommerce products\tv'
n = f(d1, n)
n = f(d2, n)
n = f(d3, n)
n = f(d4, n)
