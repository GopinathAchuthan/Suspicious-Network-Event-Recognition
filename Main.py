import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def main():
	path = "training.csv"
	a = pd.read_csv(path,delimiter="|")
	a = a.drop(columns=['alert_ids','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','score','grandparent_category','weekday','ip','ipcategory_name','ipcategory_scope','parent_category',
						'timestamp_dist','start_hour','start_minute','start_second','thrcnt_month','thrcnt_week','thrcnt_day'])

	col_str = ['client_code','categoryname','dstipcategory_dominate','srcipcategory_dominate']
	print(len(a))
	#Converting non-numeric data into numeric data
	for i in col_str:
		label_enc = LabelEncoder()
		label_enc.fit(a[i])
		a[i] = label_enc.transform(a[i])

	X = a.drop(["notified"] , axis=1) #11829
	y = a["notified"]
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)

	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	#Random Forest Classifier
	print("Random Forest Classifier")
	rfc = RandomForestClassifier(n_estimators=200)
	rfc.fit(X_train,y_train)
	pred_rfc = rfc.predict(X_test)
	print(classification_report(y_test,pred_rfc))
	print(confusion_matrix(y_test,pred_rfc))

	#SVM Classifier
	print("SVM Classifier")
	svmc = svm.SVC()
	svmc.fit(X_train,y_train)
	pred_svmc =  svmc.predict(X_test)
	print(classification_report(y_test,pred_svmc))
	print(confusion_matrix(y_test,pred_svmc))	

	#Neural Network
	print("Neural Network")
	mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500)
	mlpc.fit(X_train,y_train)
	pred_mlpc =  mlpc.predict(X_test)
	print(classification_report(y_test,pred_mlpc))
	print(confusion_matrix(y_test,pred_mlpc))

if __name__ == '__main__':
	main()