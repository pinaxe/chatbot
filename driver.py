from reduction import *
reduction = Reduction()
text = open('filename.txt').read()
reduction_ratio = 0.2
reduced_text = reduction.reduce(text, reduction_ratio)
print(reduced_text)
