import argparse
import csv  
import matplotlib.pyplot as plt
import os
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True,
  help="path to csv file")
args = vars(ap.parse_args())

train_iou=[]
train_loss=[]
val_iou=[]
val_loss=[]
with open(os.path.abspath(args['file']), 'r') as csvfile:
  plotData = csv.reader(csvfile, delimiter=',')
  next(plotData)
  for row in plotData:
    train_loss.append(round(float(row[1]), 4))
    val_loss.append(round(float(row[2]), 4))


# ## Learning curves

# In[53]:


plt.figure('Loss', figsize=(10, 6))
# plt.subplot(211)
plt.plot(train_loss, label='Train loss')
plt.plot(val_loss, label='Val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
