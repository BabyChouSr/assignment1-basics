cat hello_batch.py  
import time

print('Waiting')
time.sleep(30)
print('Done waiting')

with open('output.txt') as out:
   out.write('hello\n')

