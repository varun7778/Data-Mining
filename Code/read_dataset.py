import numpy as np
import sys
sys.path.append('/media/rahul/Stuff/github/rtg8055/Breast-Cancer-Detection/Scripts')
import decision_tree as dt
import Evaluation as ev
import knn as knn
import SVM as svm
import matplotlib.pyplot as plt

'''
   1. Sample code number           
   2. Clump Thickness 
   3. Uniformity of Cell Size      
   4. Uniformity of Cell Shape     
   5. Marginal Adhesion            
   6. Single Epithelial Cell Size  
   7. Bare Nuclei 
   8. Bland Chromatin              
   9. Normal Nucleoli              
  10. "Mi,toses"  
  11. Class:                       
 '''



my_data = []
with open("../Data/breast-cancer-wisconsin-data.csv") as file:
  for line in file:
    if(line.find('?') != -1):
      continue
    line = line.rstrip().split(',')[1:]
    line = [int(l) for l in line]
    line[-1] = str(line[-1])

    my_data.append(line)

#splitting 30% and 70%
total = len(my_data)
train = int(.7 *total)
test =total - train
print(total,train,test)

train_data = my_data[0:train]
test_data = my_data[train:]


tree= dt.buildtree(train_data)
expected_results = [row[-1] for row in test_data]
predicted_results = dt.predict(test_data,tree)
# print(predicted_results,expected_results)


predicted_results2 = knn.predict(train_data,test_data)
predicted_results3 = svm.predict(train_data,test_data)

print("\n----Confusion Matrix----\n")

'''
      predicted 
        2   4
actual 2 a    b
       4 c    d

'''

c1 = ev.confusion_matrix(predicted_results,expected_results)
c2 = ev.confusion_matrix(predicted_results2,expected_results)
c3 = ev.confusion_matrix(predicted_results3,expected_results)
print("Decision Tree")
print(c1)
print("\nK nearest Neighbours")
print(c2)
print("\nSVM")
print(c3)


print("\n----Kappa Score----\n")
print("Decision Tree")
print(ev.kappa_score(predicted_results,expected_results))
print("\nK nearest Neighbours")
print(ev.kappa_score(predicted_results2,expected_results))
print("\nSVM")
print(ev.kappa_score(predicted_results3,expected_results))
print("\n----Mean Absolute Error----\n")
ma1 = ev.MAE(predicted_results,expected_results)
ma2 = ev.MAE(predicted_results2,expected_results)
ma3 = ev.MAE(predicted_results3,expected_results)
print("Decision Tree")
print ma1
print("\nK nearest Neighbours")
print ma2
print("\nSVM")
print ma3


print("\n----Precision and Recall----\n")
print("Decision Tree")
print(ev.precision_recall(predicted_results,expected_results))
print("\nK nearest Neighbours")
print(ev.precision_recall(predicted_results2,expected_results))
print("\nSVM")
print(ev.precision_recall(predicted_results3,expected_results))



acc1= ev.accuracy(predicted_results,expected_results)
acc2= ev.accuracy(predicted_results2,expected_results)
acc3= ev.accuracy(predicted_results3,expected_results)
print("Decision Tree")
print acc1
print("\nK nearest Neighbours")
print acc2
print("\nSVM")
print acc3

fig ,axes = plt.subplots(2,2)

plt.tight_layout()
axes[0,0].plot([1,2,3],[c1[0][0],c2[0][0],c3[0][0]])
axes[0,0].set_title("Correctly Classified instances")
axes[0,1].plot([1,2,3],[c1[1][1],c2[1][1],c3[1][1]])
axes[0,1].set_title("Incorrectly Classified instances")
axes[1,0].plot([1,2,3],[acc1,acc2,acc3])
axes[1,0].set_title("Accuracy")
axes[1,1].plot([1,2,3],[ma1,ma2,ma3])
axes[1,1].set_title("Mean Absolute Error")

plt.show()




