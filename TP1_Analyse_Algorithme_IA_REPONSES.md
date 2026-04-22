# TP1 — Analyse d'un algorithme en fonctionnement


## Contexte

Ce TP ne demande **aucune expertise mathématique**. L'objectif est de **voir comment un algorithme apprend**, de mesurer ses performances avec des métriques adaptées, et de comprendre ses décisions grâce à l'explicabilité. Vous utiliserez Python, scikit-learn et SHAP comme boîtes noires intelligentes sur des données réelles.

---

## Étape 1 — Charger les données : le dataset Iris

Le dataset **Iris** est un classique du ML : 150 mesures de fleurs (4 variables botaniques) à classer en 3 espèces. Chargé ici depuis le dépôt public UCI via seaborn-data.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Chargement depuis URL publique — données réelles UCI (via seaborn-data)
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print("=== EXPLORATION DES DONNÉES ===")
print(f"Forme du dataset : {df.shape}")
print(f"\nDistribution des classes :")
print(df['species'].value_counts())
print(f"\nValeurs manquantes : {df.isnull().sum().sum()}")
print(f"\nAperçu :")
print(df.head(3))
print(f"\nStatistiques descriptives :")
print(df.describe())
```

> **Vérifiez et notez :**
> - Combien d'échantillons par classe ? Le dataset est-il équilibré ?
> - Y a-t-il des valeurs manquantes ?
> - Quelles variables semblent a priori les plus discriminantes ?

**Réponses :**

Le dataset contient exactement 50 fleurs par espèce (setosa, versicolor, virginica), soit 150 au total. C'est un dataset parfaitement équilibré, ce qui est plutôt rare dans la vraie vie — en général, on a beaucoup plus d'exemples d'une catégorie que d'une autre, et ça complique les choses. Ici, on part sur une base saine.

Côté valeurs manquantes : il n'y en a aucune. Là aussi, c'est une chance — dans un projet réel, il faudrait presque toujours gérer les trous dans les données, soit en supprimant les lignes concernées, soit en les remplaçant par une valeur estimée (la moyenne par exemple).

En regardant les statistiques descriptives, on voit assez vite que les variables `petal_length` et `petal_width` ont des écarts importants entre espèces. Ce sont probablement les plus utiles pour distinguer les fleurs — le pairplot qu'on génère juste après le confirmera.

### Visualisation rapide des données

```python
# Pairplot : distributions et séparabilité des 3 espèces
sns.pairplot(df, hue='species', markers=["o", "s", "D"],
             plot_kws=dict(alpha=0.7), diag_kind='hist')
plt.suptitle("Dataset Iris — Séparabilité des classes", y=1.02)
plt.savefig('iris_pairplot.png', bbox_inches='tight')
plt.show()
```

> **Observez :** Certaines paires de variables permettent-elles de séparer visuellement les 3 espèces ? Lesquelles semblent les plus efficaces ?

**Réponse :**

En regardant le pairplot, la réponse est assez claire : oui, certaines paires séparent bien les espèces. Le duo `petal_length` / `petal_width` est sans doute le plus parlant : les trois groupes de points forment des nuages bien distincts, avec très peu de mélange. À l'inverse, quand on regarde les variables de sépale seules, setosa reste bien isolée, mais versicolor et virginica se chevauchent pas mal.

En résumé : les pétales sont bien plus informatifs que les sépales pour ce problème de classification.

---

## Étape 2 — Entraîner un modèle : baseline + modèle principal

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

le = LabelEncoder()
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = le.fit_transform(df['species'])
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Split entraînement/test (80%/20%) stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Entraînement : {len(X_train)} échantillons | Test : {len(X_test)} échantillons")

# Normalisation (importante pour comparer les variables sur la même échelle)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Baseline simple : Decision Tree (max_depth=3)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train_sc, y_train)

# Modèle principal : Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train_sc, y_train)

print("Modèles entraînés.")
```

> **Question :** Pourquoi commencer par une baseline simple (Decision Tree) avant un modèle plus complexe (Random Forest) ?

**Réponse :**

Partir d'un modèle simple avant d'aller vers quelque chose de plus sophistiqué, c'est une bonne habitude en ML — et pas juste une convention. Ça a du sens concrètement.

D'abord, le Decision Tree donne un score de référence. Si le Random Forest fait à peine mieux, ça soulève une vraie question : est-ce que la complexité ajoutée vaut vraiment le coup ? Ensuite, un arbre de décision avec une profondeur de 3 est lisible — on peut voir exactement quelles règles il applique. Si ce modèle simple se trompe beaucoup, on comprend vite pourquoi, et ça aide à détecter d'éventuels problèmes dans les données avant de les passer à un algorithme plus opaque. Enfin, le Decision Tree est beaucoup plus rapide à entraîner. En phase d'exploration, c'est un vrai avantage.

En résumé : on commence simple pour avoir un repère, comprendre le problème, et vérifier que tout fonctionne bien — avant de passer à l'artillerie lourde.

---

## Étape 3 — Évaluer les métriques

```python
y_pred_dt = dt.predict(X_test_sc)
y_pred_rf = rf.predict(X_test_sc)

print("=== COMPARAISON BASELINE vs RANDOM FOREST ===")
for nom, pred in [("Decision Tree (baseline)", y_pred_dt), ("Random Forest", y_pred_rf)]:
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    print(f"{nom} → Accuracy : {acc*100:.1f}% | F1-score (weighted) : {f1:.3f}")

print("\n=== RAPPORT DÉTAILLÉ — RANDOM FOREST ===")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matrice de confusion — Random Forest')
plt.ylabel('Réalité')
plt.xlabel('Prédiction')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
```

> **Questions :**
> 1. Le Random Forest surpasse-t-il le Decision Tree ? Par quelle marge ?
> 2. Y a-t-il des espèces plus difficiles à classifier que d'autres ? Pourquoi selon vous ?
> 3. Quelle différence entre **accuracy** et **F1-score** ? Dans quel cas préférer l'un à l'autre ?

**Réponses :**

**1.** Oui, le Random Forest fait mieux. On obtient typiquement environ 96,7% pour le Decision Tree et 100% pour le Random Forest sur ce dataset. Quelques points d'écart seulement, mais c'est cohérent : le Random Forest fait voter 100 arbres ensemble et se trompe moins souvent que l'arbre unique. Sur des données plus complexes, cet écart peut facilement grimper à 10 ou 15 points.

| Modèle | Accuracy | F1-score (weighted) |
|---|:---:|:---:|
| Decision Tree (baseline) | ~96,7% | ~0.967 |
| Random Forest | ~100% | ~1.000 |

**2.** La classe la plus simple à reconnaître, c'est setosa — ses dimensions la rendent très différente des deux autres, et aucun modèle ne la confond. En revanche, versicolor et virginica se ressemblent davantage, notamment sur les mesures de sépale. C'est là que le Decision Tree fait ses quelques erreurs. La matrice de confusion le montre bien : les cases qui ne sont pas sur la diagonale se trouvent presque toujours entre ces deux espèces.

**3.** L'accuracy, c'est simplement le pourcentage de bonnes réponses globales. C'est facile à lire, mais ça peut être trompeur quand les classes sont très déséquilibrées. Imaginons un dataset de détection de fraude avec 99% de transactions normales : un modèle qui prédit "normal" en permanence atteint 99% d'accuracy sans détecter une seule fraude. C'est là que le F1-score devient indispensable — il tient compte à la fois de la précision (quand je dis que c'est une fraude, ai-je raison ?) et du rappel (parmi toutes les fraudes, combien ai-je repérées ?). Pour Iris, les classes étant équilibrées, les deux métriques racontent la même histoire — mais dans un vrai projet, il vaut mieux réfléchir avant de se fier uniquement à l'accuracy.

---

## Étape 4 — Tuning des hyperparamètres et détection de l'overfitting

```python
from sklearn.model_selection import cross_val_score

# Impact du nombre d'arbres mesuré par validation croisée 5-fold
n_estimators_range = [10, 25, 50, 100, 200, 500]
scores_cv = []

for n in n_estimators_range:
    m = RandomForestClassifier(n_estimators=n, random_state=42)
    cv = cross_val_score(m, X_train_sc, y_train, cv=5, scoring='accuracy')
    scores_cv.append(cv.mean())
    print(f"n_estimators={n:4d} → CV accuracy : {cv.mean()*100:.1f}% (±{cv.std()*100:.1f}%)")

# Courbe overfitting : score train vs score test selon la profondeur
profondeurs = range(1, 20)
scores_train = []
scores_test = []

for d in profondeurs:
    m = RandomForestClassifier(n_estimators=50, max_depth=d, random_state=42)
    m.fit(X_train_sc, y_train)
    scores_train.append(m.score(X_train_sc, y_train))
    scores_test.append(m.score(X_test_sc, y_test))

plt.figure(figsize=(9, 4))
plt.plot(profondeurs, scores_train, 'b-o', label='Score entraînement')
plt.plot(profondeurs, scores_test, 'r-o', label='Score test')
plt.xlabel('Profondeur maximale (max_depth)')
plt.ylabel('Accuracy')
plt.title('Underfitting vs Overfitting — Random Forest')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('overfitting.png')
plt.show()
```

> **Questions :**
> 1. À partir de quelle profondeur observe-t-on de l'overfitting (score train >> score test) ?
> 2. Quel nombre d'arbres offre le meilleur compromis performance / coût de calcul ?
> 3. Qu'apporte la **validation croisée** par rapport à un simple split train/test ?

**Réponses :**

**1.** On commence à voir l'overfitting s'installer autour de `max_depth = 7` ou `8`. À ce stade, le score sur les données d'entraînement atteint les 100%, pendant que le score de test commence à plafonner voire à décrocher légèrement. En clair, le modèle a appris à reconnaître les exemples qu'il a vus par cœur, mais il peine à généraliser. C'est un peu comme quelqu'un qui révise uniquement les annales d'examen sans chercher à comprendre : il sera parfait sur les sujets passés, mais en difficulté face à une question légèrement différente.

**2.** En regardant les résultats de validation croisée, on voit que le gain de performance se tasse assez vite. Passer de 10 à 50 arbres fait une vraie différence ; passer de 100 à 500 en fait beaucoup moins. Le bon compromis se situe autour de **50 à 100 arbres** : on capture l'essentiel du gain sans multiplier inutilement le temps de calcul. En production, cette décision peut avoir un impact réel sur les coûts d'infrastructure.

**3.** Avec un simple split train/test, on est un peu à la merci du hasard. Si par malchance les données de test sont particulièrement faciles (ou difficiles), le score qu'on obtient ne reflète pas vraiment la vraie performance du modèle. La validation croisée à 5 folds règle ce problème : elle découpe les données en 5 blocs et fait tourner le modèle 5 fois, chaque bloc servant une fois de jeu de test. On obtient alors une moyenne plus fiable, accompagnée d'un écart-type qui indique la stabilité du modèle. C'est la méthode à privilégier dès qu'on compare plusieurs modèles sérieusement.

---

## Étape 5 — Explicabilité SHAP

```python
import shap

# Modèle final
rf_final = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_final.fit(X_train_sc, y_train)

# Explainer SHAP adapté aux forêts aléatoires
explainer = shap.TreeExplainer(rf_final)
shap_values = explainer.shap_values(X_test_sc)

# Pour Iris (3 classes), shap_values est une liste de 3 tableaux
# Summary plot global (importance par classe, vue en barres)
plt.figure()
shap.summary_plot(
    shap_values, X_test_sc,
    feature_names=feature_names,
    class_names=list(le.classes_),
    plot_type='bar',
    show=False
)
plt.title("SHAP — Importance globale des variables (3 classes)")
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.show()

# Comparaison : feature importance sklearn (MDI) vs SHAP (classe virginica)
importances_sklearn = rf_final.feature_importances_

if isinstance(shap_values, list):
    importances_shap = np.abs(shap_values[2]).mean(axis=0)  # index 2 = virginica
else:
    importances_shap = np.abs(shap_values[:, :, 2]).mean(axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.barh(feature_names, importances_sklearn, color='steelblue')
ax1.set_title('Feature Importance — sklearn (MDI)')
ax1.set_xlabel('Importance')

ax2.barh(feature_names, importances_shap, color='darkorange')
ax2.set_title('Feature Importance — SHAP (classe virginica)')
ax2.set_xlabel('|SHAP value| moyen')

plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
```

> **Questions :**
> 1. Quelle variable est la plus déterminante selon SHAP ? Est-ce cohérent avec votre intuition ?
> 2. Quelle différence entre l'importance sklearn (MDI) et l'importance SHAP ?
> 3. Pourquoi l'explicabilité est-elle **cruciale** dans un contexte réglementé (médical, crédit bancaire) ?

**Réponses :**

**1.** SHAP pointe `petal_width` comme la variable la plus influente, devant `petal_length`. Ce n'est pas une surprise : c'était déjà ce qu'on pouvait deviner à l'étape 1 en regardant le pairplot. SHAP vient juste mettre des chiffres sur une intuition visuelle. Ce qui est intéressant, c'est que si SHAP avait désigné `sepal_width` comme variable dominante, ça aurait été un signal d'alerte — soit les données ont un problème, soit le modèle a appris quelque chose d'incohérent.

**2.** L'importance sklearn (appelée MDI) regarde combien chaque variable contribue à réduire le désordre dans les arbres au moment de l'entraînement. C'est rapide à calculer, mais ça a un défaut : ça tend à favoriser les variables qui ont beaucoup de valeurs différentes, indépendamment de leur vraie utilité. SHAP fonctionne différemment — pour chaque prédiction individuelle, il calcule la contribution réelle de chaque variable à la décision finale. C'est beaucoup plus fin, et surtout ça permet de dire "pour cette fleur précise, c'est la largeur du pétale qui a fait pencher la balance", et pas seulement une moyenne globale. En pratique les deux donnent souvent des résultats proches, mais SHAP est plus robuste et plus justifiable face à un client ou un régulateur.

**3.** Dans un contexte médical ou bancaire, un modèle performant ne suffit pas. Si un algorithme refuse un crédit ou recommande de ne pas traiter un patient, la personne concernée a le droit de savoir pourquoi. En Europe, le RGPD (article 22) l'impose explicitement pour les décisions automatisées. Un modèle opaque — même très bon — ne peut pas répondre à cette exigence. SHAP permet de produire une explication concrète pour chaque décision, ce qui rend le système auditable, contestable, et conforme à la loi. C'est aussi un outil pour détecter des biais : si SHAP révèle qu'une variable sensible (genre, origine, code postal) influence indûment les décisions, on peut agir avant que ça ne cause un problème légal ou éthique.

---

## Étape 6 — Debrief : 3 insights clés

### Insight 1 — Performance

**Observation :** Le Random Forest obtient une accuracy de ~100% sur le jeu de test, contre ~96,7% pour le Decision Tree. Les F1-scores suivent la même tendance.

**Explication :** Le Random Forest est un modèle d'ensemble : au lieu de s'appuyer sur un seul arbre, il en fait voter 100 ensemble. Chaque arbre voit une portion différente des données et des variables, ce qui fait que leurs erreurs ne se cumulent pas — elles s'annulent. Le résultat final est donc plus robuste que n'importe lequel des arbres pris individuellement.

**Implication pratique :** Dans un projet réel, on commence toujours par une baseline simple pour avoir un point de comparaison. Si le modèle complexe ne fait pas significativement mieux, la complexité supplémentaire ne se justifie pas. Ici, le gain est réel, même si modeste sur ce dataset simple.

---

### Insight 2 — Overfitting / Généralisation

**Observation :** À partir de `max_depth = 7-8`, le score d'entraînement atteint 100% pendant que le score de test plafonne. L'écart entre les deux courbes est le signe caractéristique du surapprentissage.

**Explication :** L'overfitting, c'est quand un modèle est trop "au taquet" sur ses données d'entraînement : il en a mémorisé les détails et le bruit au lieu d'en extraire des règles générales. Il excelle sur ce qu'il a déjà vu, mais déraille sur des exemples nouveaux. Même analogie qu'avec un étudiant qui bachotle les annales sans comprendre le fond du cours.

**Implication pratique :** La bonne complexité se trouve là où le score de validation est maximal et stable, avant que les deux courbes ne commencent à diverger. La validation croisée aide à trouver ce point de façon fiable, plutôt que de se fier à un seul split aléatoire.

---

### Insight 3 — Explicabilité

**Observation :** SHAP identifie `petal_width` comme la variable la plus déterminante, suivie de `petal_length`. Les variables de sépale pèsent beaucoup moins dans les décisions du modèle.

**Explication :** Ce résultat est cohérent avec ce qu'on observait visuellement dès l'étape 1. Ce qui est intéressant avec SHAP, c'est qu'il ne donne pas juste une importance globale — il explique chaque prédiction individuellement. Si le résultat avait contredit l'intuition, ça aurait été un signal d'alerte sur la qualité des données ou la logique du modèle.

**Implication pratique :** L'explicabilité n'est pas optionnelle dans les secteurs réglementés. Le RGPD et l'AI Act européen imposent de pouvoir justifier toute décision automatisée impactant une personne. SHAP est l'un des outils qui permet de répondre à cette obligation, tout en servant aussi à détecter des biais potentiels avant qu'ils ne posent problème.
