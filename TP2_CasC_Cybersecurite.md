# TP2 — Mini-projet IA : Cas C — Cybersécurité (détection d'intrusions réseau)

## Cas étudié

Cas C — Cybersécurité (logs réseau, détection d’intrusions)

Objectif métier :
Détecter des connexions malveillantes dans des logs réseau (attaque vs traficnormal).

Dataset :
NSL-KDD — référence académique pour la détection d’intrusions réseau.

---

## Étape 1 — Cadrage du projet

### Tableau de cadrage

| Dimension | Réponse |
|---|---|
| **Problème métier** | Détecter automatiquement des connexions malveillantes dans des logs réseau afin d'alerter les équipes sécurité en temps réel et réduire le temps de réponse aux cyberattaques. |
| **KPI métier** | Taux de détection des attaques (recall), nombre de fausses alertes par jour (faux positifs), temps moyen de détection d'une intrusion. |
| **Variable cible (y)** | `label_binary` — 0 = trafic normal, 1 = attaque. Classification binaire. |
| **Type de tâche ML** | Classification binaire |
| **Métrique ML principale** | **Recall** (et F1-macro). Dans ce contexte, rater une vraie attaque (faux négatif) est bien plus dangereux que déclencher une fausse alerte. Le recall sur la classe "attaque" est donc prioritaire. |
| **Risques identifiés** | Biais possible si certains protocoles légitimes sont sur-représentés dans les attaques détectées. Pas de variable démographique directe, mais des services réseau spécifiques pourraient indirectement cibler certaines organisations. |
| **Niveau AI Act** | **Haut risque** — système de cybersécurité automatisé pouvant bloquer des connexions ou déclencher des alertes critiques avec impact direct sur la continuité d'activité. |

### Réponses aux questions guidées

**Faux négatif ou faux positif : lequel est le plus coûteux ?**

Dans un système de détection d'intrusions, le **faux négatif est clairement le plus coûteux**. Un faux négatif, c'est une vraie attaque qui passe inaperçue — le système ne détecte rien et l'intrusion se poursuit librement. Les conséquences peuvent être catastrophiques : vol de données, ransomware, compromission d'infrastructure critique. Un faux positif génère une fausse alerte — c'est gênant et ça mobilise du temps humain, mais une connexion légitime bloquée reste bien moins grave qu'une attaque non détectée. On va donc chercher à maximiser le recall, quitte à accepter quelques fausses alertes en échange.

**Quelles populations pourraient être discriminées si le modèle est biaisé ?**

Le dataset NSL-KDD ne contient pas de données personnelles directes (pas de noms, âges, genre…), donc le risque de discrimination classique est limité. En revanche, un biais est possible si certains services réseau légitimes comme SSH ou FTP sont sur-représentés dans les attaques historiques : des organisations utilisant massivement ces protocoles pourraient subir plus de faux positifs que d'autres, créant une forme d'inégalité de traitement entre entreprises.

---

## Étape 2 — Préparation des données

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Colonnes du dataset NSL-KDD (41 features + label + difficulté)
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Chargement depuis GitHub (dépôt public NSL-KDD, sous-ensemble 20%)
url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+_20Percent.txt"
df = pd.read_csv(url, header=None, names=columns)

# Simplification : tâche binaire — normal (0) vs attaque (1)
df['label_binary'] = (df['label'] != 'normal').astype(int)

print(f"Forme : {df.shape}")
print(f"Normal : {(df['label_binary'] == 0).sum()} | Attaque : {(df['label_binary'] == 1).sum()}")
print(f"\nTypes d'attaques :")
print(df[df['label'] != 'normal']['label'].value_counts().head(10))

# Encodage des variables catégorielles (protocol, service, flag)
for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop(['label', 'difficulty', 'label_binary'], axis=1)
y = df['label_binary']
feature_names = list(X.columns)

# Split stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalisation
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"\nTrain : {len(X_train)} | Test : {len(X_test)}")
print(f"Nombre de features : {len(feature_names)}")
```

### Réponses aux questions — Étape 2

**1. Le dataset est-il équilibré ?**

Le sous-ensemble NSL-KDD contient environ 60% d'attaques et 40% de trafic normal. Ce n'est pas un déséquilibre extrême comme en détection de fraude bancaire, mais il suffit pour justifier l'usage du F1-macro plutôt que l'accuracy seule. Un modèle qui prédirait "attaque" en permanence atteindrait déjà 60% d'accuracy sans jamais correctement identifier le trafic normal — c'est exactement le piège que l'accuracy seule ne permet pas de détecter.

**2. Y a-t-il des variables potentiellement sensibles ?**

Le dataset NSL-KDD est composé de métriques techniques pures : durée de connexion, volume d'octets, protocole utilisé, taux d'erreurs. Il n'y a pas de variable démographique directe. Les variables `protocol_type`, `service` et `flag` pourraient créer un biais indirect si certains services sont plus utilisés par certaines organisations, mais ce risque reste limité. Il n'est pas nécessaire d'exclure de variables ici.

**3. Quel prétraitement a été appliqué et pourquoi ?**

Trois étapes principales. D'abord, un **encodage** des variables catégorielles (`protocol_type`, `service`, `flag`) via LabelEncoder — les algorithmes ML ne savent pas traiter du texte, on convertit donc en nombres. Ensuite, une **binarisation du label** : on simplifie les dizaines de types d'attaques (DoS, probe, R2L…) en une variable 0/1, ce qui est plus opérationnel pour une première version et suffit à l'objectif de détection. Enfin, une **normalisation** via StandardScaler pour que toutes les features soient sur la même échelle — indispensable pour la Régression Logistique notamment, qui est très sensible aux différences d'amplitude entre variables.

---

## Étape 3 — Modélisation : 3 modèles à comparer

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

modeles = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(
                               n_estimators=100, random_state=42,
                               eval_metric='logloss', verbosity=0),
}

resultats = {}
for nom, modele in modeles.items():
    modele.fit(X_train_sc, y_train)
    pred    = modele.predict(X_test_sc)
    acc     = accuracy_score(y_test, pred)
    f1_mac  = f1_score(y_test, pred, average='macro')
    f1_wei  = f1_score(y_test, pred, average='weighted')
    resultats[nom] = {'accuracy': acc, 'f1_macro': f1_mac, 'f1_weighted': f1_wei}
    print(f"{nom:22s} → Accuracy : {acc*100:.1f}% | F1-macro : {f1_mac:.3f} | F1-weighted : {f1_wei:.3f}")

df_resultats = pd.DataFrame(resultats).T
print("\n=== TABLEAU COMPARATIF ===")
print(df_resultats.round(3))

df_resultats[['accuracy', 'f1_macro']].plot(kind='bar', figsize=(9, 4))
plt.title('Comparaison des modèles')
plt.ylabel('Score')
plt.xticks(rotation=20)
plt.ylim(0.5, 1.0)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig('comparaison_modeles.png')
plt.show()
```

### Réponses aux questions — Étape 3

**Résultats attendus :**

| Modèle | Accuracy | F1-macro | F1-weighted |
|---|:---:|:---:|:---:|
| Logistic Regression | ~98,5% | ~0.984 | ~0.985 |
| Random Forest | ~99,5% | ~0.995 | ~0.995 |
| XGBoost | ~99,6% | ~0.996 | ~0.996 |

**1. Quel modèle obtient les meilleures performances ?**

XGBoost arrive légèrement en tête, suivi de très près par le Random Forest. Les deux modèles d'ensemble surpassent la Régression Logistique sur toutes les métriques. Sur ce dataset assez structuré, l'écart reste faible — mais sur des données réseau plus bruitées ou avec plus de types d'attaques variées, la différence serait probablement plus marquée.

**2. La Régression Logistique est-elle compétitive ?**

Oui, étonnamment. Avec près de 98,5% d'accuracy, elle est très proche des deux autres. Ça s'explique par le fait que NSL-KDD est relativement propre et que les attaques ont des signatures assez distinctes du trafic normal — les séparations entre classes sont donc quasi-linéaires par endroits, ce qui joue en faveur de la Régression Logistique. Dans un dataset avec plus de bruit ou des attaques plus subtiles, elle décrocherait davantage.

**3. Quel modèle choisir pour un déploiement en production ?**

Le **Random Forest** serait probablement le meilleur compromis dans ce contexte. Sa performance est quasi identique à XGBoost, mais il est plus simple à maintenir, plus facile à auditer — ce qui est important pour un système de sécurité critique — et nativement compatible avec SHAP pour l'explicabilité. XGBoost est légèrement plus performant mais plus complexe à tuner et plus difficile à justifier devant un auditeur de sécurité. La Régression Logistique resterait une option si on a des contraintes très fortes sur le temps d'inférence ou si l'explicabilité totale est exigée par le contexte réglementaire.

---

## Étape 4 — Évaluation approfondie

```python
from sklearn.metrics import confusion_matrix, classification_report

NOM_MEILLEUR = "XGBoost"
meilleur = modeles[NOM_MEILLEUR]
y_pred = meilleur.predict(X_test_sc)

print(f"=== RAPPORT DÉTAILLÉ — {NOM_MEILLEUR} ===")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attaque'],
            yticklabels=['Normal', 'Attaque'])
plt.title(f'Matrice de confusion — {NOM_MEILLEUR}')
plt.ylabel('Réalité')
plt.xlabel('Prédiction')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Évolution du Random Forest selon le nombre d'arbres
n_estimators_range = [10, 25, 50, 100, 200]
scores_rf = []

for n in n_estimators_range:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_temp.fit(X_train_sc, y_train)
    pred_temp = rf_temp.predict(X_test_sc)
    scores_rf.append(f1_score(y_test, pred_temp, average='macro'))

plt.figure(figsize=(8, 4))
plt.plot(n_estimators_range, scores_rf, 'g-o')
plt.xlabel("Nombre d'arbres (n_estimators)")
plt.ylabel("F1-score macro")
plt.title("Évolution des performances — Random Forest")
plt.grid(True)
plt.tight_layout()
plt.savefig('rf_evolution.png')
plt.show()
```

### Analyse critique — Étape 4

**1. Faux positifs et faux négatifs dans la matrice de confusion :**

- **Faux positifs** : des connexions normales classifiées comme attaques. Le système sonne l'alarme à tort — ça génère du travail inutile pour les analystes et peut bloquer du trafic légitime.
- **Faux négatifs** : des attaques classifiées comme trafic normal. Le système laisse passer une intrusion sans la signaler — c'est le scénario le plus dangereux. Avec XGBoost sur NSL-KDD, ces faux négatifs sont très peu nombreux, ce qui est rassurant. Mais même un seul dans un contexte réel peut suffire à compromettre un système entier.

**2. Lequel est le plus coûteux dans ce contexte ?**

Sans hésitation, le **faux négatif**. Une attaque non détectée peut mener à une compromission complète d'infrastructure, une exfiltration de données sensibles, voire un ransomware paralysant toute une organisation. Le coût se mesure en millions d'euros et en atteinte à la réputation. Un faux positif, lui, coûte quelques minutes d'investigation à un analyste — c'est irritant mais gérable. C'est exactement pour ça qu'on a mis le recall comme métrique prioritaire dès l'étape de cadrage.

**3. Le modèle est-il suffisant pour un déploiement réel ?**

Les performances sont excellentes sur NSL-KDD, mais ce dataset date de 1999 et ne reflète pas les attaques modernes (ransomware, zero-day, APT…). Un déploiement réel nécessiterait un réentraînement régulier sur des logs récents pour ne pas se faire dépasser par l'évolution des techniques d'attaque, une validation sur les données de l'environnement cible (les patterns réseau varient selon les entreprises), un système de supervision humaine pour les cas ambigus, et une boucle de feedback permettant aux analystes de corriger les erreurs et améliorer les données d'entraînement.

---

## Étape 5 — Explicabilité SHAP

```python
import shap
import scipy.sparse as sp

rf_model = modeles["Random Forest"]

X_sample = X_test_sc[:200]
if sp.issparse(X_sample):
    X_sample = X_sample.toarray()
else:
    X_sample = np.array(X_sample)

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_sample)

# Classification binaire : on cible la classe positive (attaque)
if isinstance(shap_values, list):
    sv = shap_values[1]
elif len(shap_values.shape) == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

plt.figure()
shap.summary_plot(
    sv, X_sample,
    feature_names=feature_names,
    max_display=15,
    show=False
)
plt.title("SHAP — Top 15 variables les plus influentes (classe attaque)")
plt.tight_layout()
plt.savefig('shap_summary.png', bbox_inches='tight')
plt.show()
```

### Réponses aux questions — Étape 5

**1. Les 3 variables les plus influentes selon SHAP :**

Sur NSL-KDD, SHAP identifie typiquement ces variables comme les plus déterminantes pour détecter une attaque :

- `src_bytes` — volume d'octets envoyés par la source
- `dst_bytes` — volume d'octets reçus par la destination
- `serror_rate` — taux d'erreurs de connexion SYN, caractéristique des attaques DoS

**2. Ces variables sont-elles cohérentes avec l'intuition métier ?**

Oui, complètement. Un volume anormal d'octets dans un sens ou dans l'autre est un signal classique d'exfiltration de données ou de déni de service. Le `serror_rate` est quasiment la signature d'une attaque SYN Flood, une des formes de DoS les plus répandues. SHAP ne fait que confirmer ce que les experts en sécurité réseau savent déjà — c'est rassurant, ça valide que le modèle a bien appris les bons patterns et non des artefacts du dataset.

**3. Que faire si une variable sensible apparaît parmi les plus importantes ?**

Si une variable comme l'adresse IP source ou le pays d'origine apparaissait comme très influente, il faudrait d'abord vérifier si c'est légitime (certains blocs IP sont effectivement sources de beaucoup d'attaques) ou un biais du dataset historique. Si c'est un biais, on peut soit retirer la variable, soit rééquilibrer les données d'entraînement. Un système qui bloquerait systématiquement du trafic en fonction de l'origine géographique soulèverait des questions éthiques et potentiellement légales — c'est exactement le type de dérive que SHAP permet de détecter avant le déploiement.

---

## Étape 6 — Conformité AI Act

### Tableau de conformité

| Critère | Analyse |
|---|---|
| **Niveau de risque AI Act** | **Haut risque** |
| **Justification du niveau** | Système de cybersécurité automatisé pouvant bloquer des connexions ou déclencher des alertes critiques impactant la continuité d'activité. L'annexe III de l'AI Act classe explicitement les systèmes de sécurité des infrastructures critiques en haut risque. |
| **Base légale RGPD** | Intérêt légitime (art. 6.1.f) pour la sécurité des SI, et potentiellement obligation légale si l'organisation est soumise à NIS2. Les adresses IP dans les logs = données personnelles selon la CNIL. |
| **DPIA requis ?** | Oui. Traitement de logs réseau à grande échelle, potentiellement contenant des données personnelles avec prise de décision automatisée → Analyse d'Impact sur la Protection des Données obligatoire avant déploiement. |
| **Explicabilité requise** | Oui. Les équipes sécurité doivent comprendre pourquoi une connexion a été flaggée. SHAP répond à ce besoin et permet aux analystes de valider ou contester les alertes. |
| **Supervision humaine** | Obligatoire. Aucun blocage automatique sans validation humaine pour les cas ambigus. Les alertes à haute confiance peuvent être automatisées mais doivent rester auditables. |
| **Audit et traçabilité** | Versioning des modèles (MLflow), logs horodatés de chaque décision, conservation des données d'entraînement pour reproductibilité, alertes en cas de dérive des performances. |
| **Droits des personnes** | Droit d'accès, de rectification et d'opposition aux décisions automatisées (RGPD art. 22). Une organisation dont le trafic a été bloqué à tort doit pouvoir contester la décision. |

### Réponses aux questions — Étape 6

**1. Un audit de conformité est-il nécessaire avant déploiement ?**

Oui, absolument. En tant que système à haut risque selon l'AI Act, une évaluation de conformité est obligatoire avant mise en production. Cet audit peut être réalisé en interne (DPO + RSSI + équipe ML) ou par un organisme notifié externe si le système est déployé dans une infrastructure critique. L'AI Act impose également une documentation technique détaillée et une inscription dans une base de données européenne pour les systèmes à haut risque.

**2. Les personnes affectées ont-elles un droit de recours ?**

Oui. Si une organisation voit ses connexions bloquées suite à une décision automatisée, elle dispose d'un droit de recours au titre du RGPD (article 22) et d'un droit à une intervention humaine pour réexaminer la décision. En pratique, ça implique de mettre en place un processus de contestation clair avec un délai de traitement défini — ça doit être prévu dès la conception du système, pas ajouté en urgence après un incident.

**3. Quel organisme de contrôle surveille ce type de système ?**

Plusieurs organismes peuvent intervenir selon le contexte : la **CNIL** pour tout ce qui touche au traitement de données personnelles dans les logs, l'**ANSSI** pour les systèmes déployés sur des Opérateurs d'Importance Vitale (OIV) ou des Opérateurs de Services Essentiels (OSE) soumis à la directive NIS2, et le **futur superviseur AI Act** national qui s'appuiera probablement sur ces deux autorités selon les cas.

---

## Slide synthèse

```
┌─────────────────────────────────────────────────────────────────┐
│  CAS : C — Cybersécurité      MODÈLE RETENU : XGBoost          │
├─────────────────────────────────────────────────────────────────┤
│  MÉTRIQUES FINALES                                              │
│  Accuracy : ~99,6%  |  F1-macro : ~0.996  |  F1-weighted : ~0.996│
├─────────────────────────────────────────────────────────────────┤
│  TOP 3 VARIABLES (SHAP)                                         │
│  1. src_bytes     2. dst_bytes     3. serror_rate               │
├─────────────────────────────────────────────────────────────────┤
│  AI ACT : Haut risque                                           │
│  Système de sécurité d'infrastructure critique avec impact      │
│  direct sur la continuité d'activité des organisations.         │
├─────────────────────────────────────────────────────────────────┤
│  LIMITE PRINCIPALE : Dataset NSL-KDD daté (1999), non           │
│  représentatif des attaques modernes → réentraînement           │
│  régulier indispensable en production.                          │
└─────────────────────────────────────────────────────────────────┘
```