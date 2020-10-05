


def confusion_matrix(predicted,actual):
	a,b,c,d=0,0,0,0
	for p,s in zip(predicted,actual):
		# print p,s
		if(p==s):
			if(p=='2'):
				a+=1
			else:
				d+=1
		else:
			if(s=='2'):
				b+=1
			else:
				c+=1
	return [[a,b],[c,d]]

	'''
			predicted 
			  2   4
	actual 2 a 	  b
		   4 c    d

	'''



def MAE(predicted,actual):
	sum=0
	for a,b in zip(predicted,actual):
		if(a==b):
			continue
		else:
			sum+=1
	return float(sum) / len(predicted)

def precision_recall(predicted,actual):
	con_mat = confusion_matrix(predicted,actual)

	precision_1 = float(con_mat[0][0]) / (con_mat[0][0] + con_mat[1][0])
	precision_2 = float(con_mat[1][1]) / (con_mat[0][1] + con_mat[1][1])

	recall_1 = float(con_mat[0][0] ) / (con_mat[0][0] + con_mat[0][1])
	recall_2 = float(con_mat[1][1] ) / (con_mat[1][0] + con_mat[1][1])

	return "\nPrecision and Recall for Benign:\t\t" + str(precision_1) + "\t\t"+ str(recall_1) + "\nPrecision and Recall for Malignant: \t\t" + str(precision_2)  +"\t\t" + str(recall_2)
def kappa_score(predicted,actual):
	from sklearn.metrics import cohen_kappa_score
	return cohen_kappa_score(predicted,actual)

def accuracy(predicted,actual):
	con_mat = confusion_matrix(predicted,actual)

	return float(con_mat[0][0] + con_mat[1][1]) / (con_mat[0][0] + con_mat[1][0] + con_mat[0][1] + con_mat[1][1] )