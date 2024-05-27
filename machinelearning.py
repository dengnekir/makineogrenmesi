import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report,roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from scipy.stats import ttest_ind
import seaborn as sns

# Metin dosyasından veri okuma
file_path = "veri-seti.txt"
df = pd.read_csv(file_path, delimiter="\t")

X = df.iloc[:, :-1]  # Hedef değişken hariç tüm öznitelikler
y = df.iloc[:, -1]   # Hedef değişken (son sütun)

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Normalizasyon işlemi
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Çoklu Doğrusal Regresyon Analizi
linear_reg = LinearRegression()
linear_reg.fit(X_train_normalized, y_train)
linear_coef = linear_reg.coef_
linear_intercept = linear_reg.intercept_
linear_predictions = linear_reg.predict(X_test_normalized)
linear_mse = mean_squared_error(y_test, linear_predictions)

print("Çoklu Doğrusal Regresyon Katsayıları:", linear_coef)
print("Çoklu Doğrusal Regresyon Kesme Noktası:", linear_intercept)
print("Çoklu Doğrusal Regresyon Ortalama Kare Hatası (MSE):", linear_mse)

# Multinominal Lojistik Regresyon Analizi
logistic_reg = LogisticRegression(multi_class='multinomial', max_iter=1000)
logistic_reg.fit(X_train_normalized, y_train)
logistic_coef = logistic_reg.coef_
logistic_intercept = logistic_reg.intercept_
logistic_predictions = logistic_reg.predict(X_test_normalized)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)

print("\nMultinominal Lojistik Regresyon Katsayıları:", logistic_coef)
print("Multinominal Lojistik Regresyon Kesme Noktası:", logistic_intercept)
print("Multinominal Lojistik Regresyon Doğruluk Oranı:", logistic_accuracy)

# Eşik değeri belirleme
threshold = 0.5

# Lojistik regresyon modeli ile tahmin olasılıklarını alma
logistic_probabilities = logistic_reg.predict_proba(X_test_normalized)

# Olasılıkları eşik değeriyle sınıflara dönüştürme
y_pred_threshold = (logistic_probabilities[:, 1] >= threshold).astype(int)

# Eşik değeriyle elde edilen tahminler için doğruluk hesaplama
threshold_accuracy = accuracy_score(y_test, y_pred_threshold)

print(f"Eşik değeri {threshold} için Doğruluk Oranı:", threshold_accuracy)


# Naive Bayes Sınıflandırıcısı
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Test seti üzerinde tahminleri alın
y_pred_test_nb = nb_classifier.predict(X_test)

# Konfüzyon Matrisi
conf_matrix_nb = confusion_matrix(y_test, y_pred_test_nb)

# Hassasiyet, Özgünlük, Doğruluk, F1 Skoru
tn, fp, fn, tp = conf_matrix_nb.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
accuracy_nb = accuracy_score(y_test, y_pred_test_nb)
f1_score_nb = classification_report(y_test, y_pred_test_nb, output_dict=True)['weighted avg']['f1-score']

print("Naive Bayes Sınıflandırıcı için Hassasiyet (Sensitivity):", sensitivity)
print("Naive Bayes Sınıflandırıcı için Özgünlük (Specificity):", specificity)
print("\nNaive Bayes Sınıflandırıcı Test Seti için Konfüzyon Matrisi:")
print(conf_matrix_nb)
print("\nNaive Bayes Sınıflandırıcı Test Seti için Hassasiyet (Sensitivity):", sensitivity)
print("Naive Bayes Sınıflandırıcı Test Seti için Özgünlük (Specificity):", specificity)
print("Naive Bayes Sınıflandırıcı Test Seti için Doğruluk (Accuracy):", accuracy_nb)
print("Naive Bayes Sınıflandırıcı Test Seti için F1 Skoru:", f1_score_nb)

# Eğitim seti üzerinde tahminleri alma
y_pred_train_nb = nb_classifier.predict(X_train)

# Eğitim seti için sınıflandırma raporu
print("\nNaive Bayes Sınıflandırıcı Eğitim Seti için Sınıflandırma Raporu:")
print(classification_report(y_train, y_pred_train_nb))

# Test seti üzerinde tahminleri alma
y_pred_test_nb = nb_classifier.predict(X_test)

# Test seti için sınıflandırma raporu
print("\nNaive Bayes Sınıflandırıcı Test Seti için Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_test_nb))

# Test seti için karışıklık matrisi
print("\nNaive Bayes Sınıflandırıcı Test Seti için Karışıklık Matrisi:")
print(confusion_matrix(y_test, y_pred_test_nb))

# PCA modeli oluşturma
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_train_pca = pca.fit_transform(X_train_normalized)
X_test_pca = pca.transform(X_test_normalized)

cov_matrix = np.cov(X_train_normalized.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Özdeğerler pca:", eigenvalues)
print("Özvektörler pca:", eigenvectors)
# LDA modeli oluşturma
lda = LinearDiscriminantAnalysis(n_components=min(X_train_normalized.shape[1], len(set(y_train)) - 1))
X_train_lda = lda.fit_transform(X_train_normalized, y_train)
X_test_lda = lda.transform(X_test_normalized)
# Sınıf ortalamalarını hesaplama
class_means = np.mean(X_train_normalized, axis=0)
overall_mean = np.mean(X_train_normalized)

# Sınıf içi ve sınıf dışı dağılım matrislerini hesaplama
within_class_scatter_matrix = np.zeros((X_train_normalized.shape[1], X_train_normalized.shape[1]))
between_class_scatter_matrix = np.zeros((X_train_normalized.shape[1], X_train_normalized.shape[1]))

for c in np.unique(y_train):
    class_indices = np.where(y_train == c)
    class_data = X_train_normalized[class_indices]
    class_mean = np.mean(class_data, axis=0)
    
    # Sınıf içi dağılım matrisi
    within_class_scatter_matrix += (class_data - class_mean).T.dot(class_data - class_mean)
    
    # Sınıf dışı dağılım matrisi
    between_class_scatter_matrix += len(class_data) * (class_mean - overall_mean).reshape(-1, 1).dot((class_mean - overall_mean).reshape(1, -1))

# Özdeğer ve özvektörleri hesaplama
eigenvalues_lda, eigenvectors_lda = np.linalg.eig(np.linalg.inv(within_class_scatter_matrix).dot(between_class_scatter_matrix))

print("LDA Özdeğerler:", eigenvalues_lda)
print("LDA Özvektörler:", eigenvectors_lda)


# Multi-Layer Perceptron (MLP) Sınıflandırıcısı
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp.fit(X_train_normalized, y_train)

# Eğitim seti üzerinde tahminleri alma
y_train_pred_mlp = mlp.predict(X_train_normalized)
# Test seti üzerinde tahminleri alma
y_test_pred_mlp = mlp.predict(X_test_normalized)

# Performans metriklerini hesaplama
accuracy_train_mlp = accuracy_score(y_train, y_train_pred_mlp)
accuracy_test_mlp = accuracy_score(y_test, y_test_pred_mlp)
conf_matrix_train_mlp = confusion_matrix(y_train, y_train_pred_mlp)
conf_matrix_test_mlp = confusion_matrix(y_test, y_test_pred_mlp)
class_report_train_mlp = classification_report(y_train, y_train_pred_mlp)
class_report_test_mlp = classification_report(y_test, y_test_pred_mlp)

print("MLP Sınıflandırıcı Eğitim Seti için Doğruluk Oranı:", accuracy_train_mlp)
print("\nMLP Sınıflandırıcı Eğitim Seti için Konfüzyon Matrisi:")
print(conf_matrix_train_mlp)
print("\nMLP Sınıflandırıcı Eğitim Seti için Sınıflandırma Raporu:")
print(class_report_train_mlp)

print("MLP Sınıflandırıcı Test Seti için Doğruluk Oranı:", accuracy_test_mlp)
print("\nMLP Sınıflandırıcı Test Seti için Konfüzyon Matrisi:")
print(conf_matrix_test_mlp)
print("\nMLP Sınıflandırıcı Test Seti için Sınıflandırma Raporu:")
print(class_report_test_mlp)

# ROC eğrisi ve AUC değeri (MLP)
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, mlp.predict_proba(X_test_normalized)[:, 1])
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

# Support Vector Machines (SVM) Sınıflandırıcısı
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_normalized, y_train)

# SVM Sınıflandırıcısı (probability=True ekleyerek)
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train_normalized, y_train)

# Eğitim seti üzerinde tahminleri alma
y_train_pred_svm = svm.predict(X_train_normalized)
# Test seti üzerinde tahminleri alma
y_test_pred_svm = svm.predict(X_test_normalized)

# Performans metriklerini hesaplama
accuracy_train_svm = accuracy_score(y_train, y_train_pred_svm)
accuracy_test_svm = accuracy_score(y_test, y_test_pred_svm)
conf_matrix_train_svm = confusion_matrix(y_train, y_train_pred_svm)
conf_matrix_test_svm = confusion_matrix(y_test, y_test_pred_svm)
class_report_train_svm = classification_report(y_train, y_train_pred_svm)
class_report_test_svm = classification_report(y_test, y_test_pred_svm)

print("SVM Sınıflandırıcı Eğitim Seti için Doğruluk Oranı:", accuracy_train_svm)
print("\nSVM Sınıflandırıcı Eğitim Seti için Konfüzyon Matrisi:")
print(conf_matrix_train_svm)
print("\nSVM Sınıflandırıcı Eğitim Seti için Sınıflandırma Raporu:")
print(class_report_train_svm)

print("SVM Sınıflandırıcı Test Seti için Doğruluk Oranı:", accuracy_test_svm)
print("\nSVM Sınıflandırıcı Test Seti için Konfüzyon Matrisi:")
print(conf_matrix_test_svm)
print("\nSVM Sınıflandırıcı Test Seti için Sınıflandırma Raporu:")
print(class_report_test_svm)
# ROC eğrisi ve AUC değeri (SVM)
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm.predict_proba(X_test_normalized)[:, 1])
roc_auc_svm = auc(fpr_svm, tpr_svm)



# Performans Raporlaması
def report_performance(name, y_true_train, y_pred_train, y_true_test, y_pred_test, fpr, tpr, roc_auc):
    print(f"\n{name} Sınıflandırıcı")
    print("Eğitim Seti:")
    print(f"Doğruluk Oranı: {accuracy_score(y_true_train, y_pred_train)}")
    print("Konfüzyon Matrisi:")
    print(confusion_matrix(y_true_train, y_pred_train))
    print("Sınıflandırma Raporu:")
    print(classification_report(y_true_train, y_pred_train))
    
    print("\nTest Seti:")
    print(f"Doğruluk Oranı: {accuracy_score(y_true_test, y_pred_test)}")
    print("Konfüzyon Matrisi:")
    print(confusion_matrix(y_true_test, y_pred_test))
    print("Sınıflandırma Raporu:")
    print(classification_report(y_true_test, y_pred_test))
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {name}')
    plt.legend(loc="lower right")
    plt.show()

report_performance("MLP", y_train, y_train_pred_mlp, y_test, y_test_pred_mlp, fpr_mlp, tpr_mlp, roc_auc_mlp)
report_performance("SVM", y_train, y_train_pred_svm, y_test, y_test_pred_svm, fpr_svm, tpr_svm, roc_auc_svm)


# PCA bileşenlerinin görselleştirilmesi
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('PCA Bileşenler')
plt.scatter(x=X_train_pca[:, 0], y=X_train_pca[:, 1], c=y_train, cmap='plasma')  # Renk haritasını 'plasma' olarak değiştir
plt.xlabel('Bileşen x')
plt.ylabel('Bileşen y')
plt.colorbar(label='Sınıf')

# LDA bileşenlerinin görselleştirilmesi
plt.subplot(1, 2, 2)
plt.title('LDA Bileşenler')
plt.scatter(x=X_train_pca[:, 0], y=X_train_pca[:, 1], c=y_train, cmap='inferno')  # Renk haritasını 'inferno' olarak değiştir
plt.xlabel('Bileşen x')
plt.ylabel('Bileşen y')
plt.colorbar(label='Sınıf')

plt.tight_layout()
plt.show()

# Karar Ağacı Sınıflandırma Modeli
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Ağaç yapısını görselleştirme
plt.figure(figsize=(20,10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=True)
plt.show()
# Karar Ağacı Sınıflandırma Modeli (Budama ile)
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X_train, y_train)

# Ağaç yapısını görselleştirme
plt.figure(figsize=(20,10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=True)
plt.show()

# Test verisi için kestirim
y_pred = dt_classifier.predict(X_test)

n_components = min(X.shape[1], len(y.unique()) - 1) 
lda = LinearDiscriminantAnalysis(n_components=n_components)
X_lda = lda.fit_transform(X, y)

# K-en yakın komşuluk (KNN) Sınıflandırıcısı
knn = KNeighborsClassifier()

# En iyi k değerini belirlemek için GridSearchCV kullanma
param_grid = {'n_neighbors': range(1, 31)}
grid_search = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train_normalized, y_train)

# En iyi k değerini elde etme
best_k = grid_search.best_params_['n_neighbors']
print(f"En iyi k değeri: {best_k}")

# En iyi k değeri ile KNN modelini yeniden eğitme
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_normalized, y_train)
# Test seti üzerinde tahminleri alma
y_pred_test_knn = knn_best.predict(X_test_normalized)
# Performans metriklerini hesaplama
accuracy_knn = accuracy_score(y_test, y_pred_test_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_test_knn)
class_report_knn = classification_report(y_test, y_pred_test_knn)

# Performans metriklerini yazdırma
print("KNN Sınıflandırıcı (k={} için) Test Seti için Doğruluk Oranı:".format(best_k), accuracy_knn)
print("\nKNN Sınıflandırıcı Test Seti için Konfüzyon Matrisi:")
print(conf_matrix_knn)
print("\nKNN Sınıflandırıcı Test Seti için Sınıflandırma Raporu:")
print(class_report_knn)
# Test seti üzerinde tahminleri alma
y_pred_test_knn = knn_best.predict(X_test_normalized)

# ROC eğrisi ve AUC değeri (Naive Bayes)
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_classifier.predict_proba(X_test_normalized)[:, 1])
roc_auc_nb = auc(fpr_nb, tpr_nb)

# Performans metriklerini hesaplama-naive bayes
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Karar Ağacı Sınıflandırma Doğruluk Oranı:", accuracy)
print("\nKarar Ağacı Sınıflandırma Karmaşıklık Matrisi:\n", conf_matrix)
print("\nKarar Ağacı Sınıflandırma Sınıflandırma Raporu:\n", class_report)

# Performans metriklerini yazdırma
print("Naive Bayes Sınıflandırıcı Test Seti için Doğruluk Oranı:", accuracy_nb)
print("\nNaive Bayes Sınıflandırıcı Test Seti için Konfüzyon Matrisi:")
print(conf_matrix_nb)
print("\nNaive Bayes Sınıflandırıcı Test Seti için Sınıflandırma Raporu:")
print(class_report)


# ROC eğrisi ve AUC değeri (KNN)
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_best.predict_proba(X_test_normalized)[:, 1])
roc_auc_knn = auc(fpr_knn, tpr_knn)

# ROC eğrilerini çizme
plt.figure()
plt.plot(fpr_nb, tpr_nb, color='darkorange', lw=2, label=f'Naive Bayes ROC curve (area = {roc_auc_nb:.2f})')
plt.plot(fpr_knn, tpr_knn, color='blue', lw=2, label=f'KNN ROC curve (area = {roc_auc_knn:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.show()

# Test verisi için kestirim
y_pred = dt_classifier.predict(X_test)

# Performans metriklerini hesaplama
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Karar Ağacı Sınıflandırma Doğruluk Oranı:", accuracy)
print("\nKarar Ağacı Sınıflandırma Karmaşıklık Matrisi:\n", conf_matrix)
print("\nKarar Ağacı Sınıflandırma Sınıflandırma Raporu:\n", class_report)

# PCA için özniteliklerin önem sırasını belirleme
pca_components = pd.DataFrame(pca.components_, columns=X.columns)
print("\nPCA için özniteliklerin önem sırası:")
print(pca_components)

# LDA için özniteliklerin önem sırasını belirleme
lda_components = pd.DataFrame(lda.scalings_.T, columns=X.columns)
print("\nLDA için özniteliklerin önem sırası:")
print(lda_components)

# İki vektör arasındaki Öklid uzaklığını hesaplama
def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))

# İki vektör arasındaki Minkowski uzaklığını hesaplama
def minkowski_distance(v1, v2, p):
    return np.power(np.sum(np.power(np.abs(v1 - v2), p)), 1/p)

# Örnek vektörler
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Öklid uzaklığı
euclidean_dist = euclidean_distance(v1, v2)

# Minkowski uzaklığı (p=3 için)
p_value = 3
minkowski_dist = minkowski_distance(v1, v2, p_value)
# Öklid ve Minkowski uzaklıklarını terminale yazdırma
print(f"Öklid Uzaklığı: {euclidean_dist}")
print(f"Minkowski Uzaklığı (p={p_value}): {minkowski_dist}")
# Görselleştirme
plt.figure(figsize=(8, 6))
plt.scatter([v1[0], v2[0]], [v1[1], v2[1]], color='blue')  # Noktaları çiz
plt.plot([v1[0], v2[0]], [v1[1], v2[1]], '--', color='red')  # Noktalar arasındaki çizgiyi çiz
plt.text(v1[0], v1[1], 'V1', fontsize=12, ha='right')  # Vektörleri etiketle
plt.text(v2[0], v2[1], 'V2', fontsize=12, ha='left')
plt.title(f"Öklid Uzaklığı: {euclidean_dist:.2f}, Minkowski Uzaklığı (p={p_value}): {minkowski_dist:.2f}")
plt.xlabel('X Ekseni')
plt.ylabel('Y Ekseni')
plt.grid(True)
plt.show()
# Veri setinin ortalama ve standart sapma değerlerini hesaplama
feature_means = X.mean()
feature_std = X.std()

print("\nÖzniteliklerin Ortalamaları:")
print(feature_means)

print("\nÖzniteliklerin Standart Sapmaları:")
print(feature_std)


# İki özellik (sütun) seçme
column1 = X.iloc[:, 0]  # İlk sütun
column2 = X.iloc[:, 1]  # İkinci sütun

# t-testi uygulama
t_statistic, p_value = ttest_ind(column1, column2)

print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")

# p-value değerini yorumlama
alpha = 0.05
if p_value < alpha:
    print("İki grup arasında istatistiksel olarak anlamlı bir fark vardır (H0 reddedildi).")
else:
    print("İki grup arasında istatistiksel olarak anlamlı bir fark yoktur (H0 kabul edildi).")
# İki özellik (sütun) seçme
column1 = X.iloc[:, 0]  # İlk sütun
column2 = X.iloc[:, 1]  # İkinci sütun

# Normalizasyon
scaler = StandardScaler()
column1_normalized = scaler.fit_transform(column1.values.reshape(-1, 1)).flatten()
column2_normalized = scaler.fit_transform(column2.values.reshape(-1, 1)).flatten()

# t-testi uygulama
t_statistic, p_value = ttest_ind(column1_normalized, column2_normalized)

print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")

# p-value değerini yorumlama
alpha = 0.05
if p_value < alpha:
    print("İki grup arasında istatistiksel olarak anlamlı bir fark vardır (H0 reddedildi).")
else:
    print("İki grup arasında istatistiksel olarak anlamlı bir fark yoktur (H0 kabul edildi).")

