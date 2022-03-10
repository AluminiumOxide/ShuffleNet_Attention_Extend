import matplotlib.pyplot as plt
import numpy as np

# a 2D array with linearly increasing values on the diagonal
a = np.diag(range(25))

# plt.matshow(a,extent=(0.1,0.9,0.1,0.9))
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.title("Confusion Matrix")
# plt.show()

confusion_out = a
label_names = ["CA", "CE", "HSIL", "LSIL", "Normal"]
label_names_re = ["Normal", "LSIL", "HSIL", "CE", "CA"]
plt.matshow(confusion_out, cmap=plt.cm.Blues)  # Greens, Blues, Oranges, Reds
plt.colorbar()
for i in range(len(confusion_out)):
    for j in range(len(confusion_out)):
        infos = ('%.2f' % confusion_out[i, j])
        plt.annotate(infos, xy=(i, j), horizontalalignment='center', verticalalignment='center')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.xticks(range(len(label_names)), label_names)
plt.yticks(range(len(label_names)), label_names_re)
plt.xlim(-0.5, 4.5)
plt.ylim(-0.5, 4.5)
plt.title("Confusion Matrix")
# plt.subplots_adjust(left=0.15,right=0.95,wspace=0.25,hspace=0.25,bottom=0.15,top=0.95)
plt.savefig("Confusion_Matrix.png")
plt.show()