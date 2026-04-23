# TP3 — Deep Learning : classification automatique de produits e-commerce

## Contexte

**Mission** : Développer un système de classification automatique des images produits Zalando pour réduire le taux d'erreur de catégorisation de 12 % à moins de 5 %.

**Dataset** : Fashion-MNIST — 70 000 images 28×28 pixels en niveaux de gris, 10 catégories de vêtements/accessoires (Zalando Research, 2017).

---

## Étape 1 — Exploration des données

**1. Le catalogue est-il équilibré entre les catégories ? Quel impact sur l'entraînement ?**

Oui, Fashion-MNIST est parfaitement équilibré : chaque catégorie contient exactement 6 000 images en train et 1 000 en test, soit 10 % du total chacune. C'est une situation idéale — le modèle reçoit autant d'exemples de sneakers que de robes ou de sandales, il n'y a donc pas de risque qu'il devienne "spécialiste" d'une catégorie sur-représentée au détriment des autres. En production réelle, un déséquilibre (par exemple beaucoup plus de T-shirts que de bottines) biaiserait les prédictions vers les classes majoritaires et rendrait les métriques par catégorie essentielles à surveiller.

**2. Un pull et une chemise se ressemblent visuellement : cela pose-t-il un risque de confusion ?**

Oui, c'est un risque réel. Le pull (classe 2) et la chemise (classe 6) partagent une silhouette similaire — col, manches, forme générale du haut du corps. À 28×28 pixels, les détails fins comme les boutons, le col ou la matière sont difficilement lisibles. Un algorithme qui ne comprend pas la sémantique de ces éléments aura naturellement tendance à les confondre, et c'est d'ailleurs l'une des confusions les plus fréquentes que l'on observe dans les matrices de confusion sur ce dataset.

**3. La résolution 28×28 est-elle représentative d'un cas de production réel ?**

Non, pas du tout. Sur la vraie plateforme Zalando, les photos produits sont affichées et analysées en haute résolution — typiquement entre **512×512 et 1024×1024 pixels**, avec plusieurs vues par article (face, dos, détails). Le choix de 28×28 dans Fashion-MNIST est délibéré : il permet de travailler avec un dataset léger, rapide à entraîner, idéal pour l'apprentissage. En production réelle, on utiliserait des architectures comme ResNet50 ou EfficientNet, pré-entraînées sur ImageNet, capables d'exploiter des images haute résolution en couleur.

---

## Étape 2 — Prétraitement

**Pourquoi normalise-t-on les pixels entre 0 et 1 ? Que se passerait-il sans ?**

Un réseau de neurones apprend en ajustant ses poids via la descente de gradient. Si les valeurs d'entrée sont comprises entre 0 et 255, les gradients calculés sont beaucoup plus grands que si elles sont entre 0 et 1 — ce qui provoque des oscillations lors de l'entraînement, une convergence lente, voire une instabilité numérique (le phénomène dit d'*exploding gradients*). La normalisation ramène toutes les features sur une même échelle, ce qui stabilise les mises à jour de poids, accélère la convergence et permet à l'optimiseur Adam de fonctionner dans des conditions optimales. Sans cette étape, le modèle apprendrait quand même, mais beaucoup moins efficacement et avec un risque accru de ne jamais converger correctement.

---

## Étape 3 — Baseline Random Forest

**1. Cette accuracy suffit-elle pour atteindre l'objectif business (taux d'erreur < 5 %) ?**

Non. Random Forest sur Fashion-MNIST atteint typiquement **~87 % d'accuracy**, ce qui correspond à un **taux d'erreur d'environ 13 %** — soit à peine mieux que la situation initiale des 12 % d'erreurs de catalogage. L'objectif business de moins de 5 % n'est pas atteint. Ce résultat justifie à lui seul d'aller vers des architectures plus sophistiquées.

**2. Random Forest comprend-il que les pixels voisins forment des contours ou des textures ?**

Non, et c'est sa limite fondamentale sur les images. Random Forest traite chaque pixel comme une feature indépendante parmi les 784 disponibles. Il ne sait pas que le pixel (14, 10) est voisin du pixel (14, 11) et que ensemble, ils forment le bord d'une manche. Il n'a aucune notion de structure spatiale, de symétrie, ni de translation. Concrètement, si on permutait aléatoirement tous les pixels d'une image, le Random Forest obtiendrait exactement le même score — ce qui montre bien qu'il n'exploite pas du tout la structure visuelle de l'image.

---

## Étape 4 — Réseau de neurones dense (MLP)

**1. Combien de paramètres entraînables au total ?**

| Couche | Calcul | Paramètres |
|---|---|---|
| Dense(128, relu) | 784×128 + 128 | 100 480 |
| Dense(64, relu) | 128×64 + 64 | 8 256 |
| Dense(10, softmax) | 64×10 + 10 | 650 |
| **Total** | | **109 386** |

Pour comparaison, une image ne contient que 784 pixels. Le réseau utilise donc **~140 fois plus de paramètres** que de pixels d'entrée — ce qui illustre la capacité d'apprentissage du MLP, mais aussi le risque d'overfitting si les données ne sont pas suffisantes.

**2. Que fait `softmax` sur la dernière couche ?**

`softmax` transforme les scores bruts (appelés *logits*) de la dernière couche en une **distribution de probabilités** : chaque valeur est comprise entre 0 et 1, et la somme des 10 sorties vaut exactement 1. Concrètement, la sortie du modèle ressemble à : `T-shirt: 0.02, Pantalon: 0.01, ..., Sneaker: 0.89, ...`. C'est exactement ce qu'on veut pour une classification multi-classe — pouvoir lire directement la confiance du modèle pour chaque catégorie. `softmax` est l'équivalent multi-classe de la fonction `sigmoid` utilisée en classification binaire.

**3. Pourquoi `sparse_categorical_crossentropy` et pas `binary_crossentropy` ?**

`binary_crossentropy` est réservée aux problèmes à **deux classes** (0 ou 1). Ici on a 10 catégories, donc on a besoin d'une loss adaptée au multi-classe. `sparse_categorical_crossentropy` accepte les labels sous forme d'entiers (0, 1, 2... 9), ce qui correspond exactement au format de Fashion-MNIST. L'alternative serait `categorical_crossentropy` qui exige des labels en *one-hot encoding* (vecteurs binaires de taille 10) — les deux donnent le même résultat, mais `sparse_` évite de convertir les labels au préalable et consomme moins de mémoire.

---

## Étape 5 — Réseau convolutif (CNN)

**1. Pourquoi la sortie de `Conv2D(32, 3×3)` sur une image 28×28 est-elle 26×26 ?**

Un filtre 3×3 est centré sur chaque pixel pour calculer sa sortie. Sur les bords de l'image, le filtre déborderait à l'extérieur — ce qui n'est pas défini. Par défaut (sans *padding*), TensorFlow supprime donc les pixels de bord pour lesquels le calcul est impossible. On perd 1 pixel de chaque côté, soit 2 pixels par dimension : 28 - 2 = **26**. Pour conserver la taille originale, on pourrait utiliser `padding='same'`, qui ajoute des zéros autour de l'image avant d'appliquer le filtre.

**2. Quel est l'intérêt du `MaxPooling2D` ? Que se passerait-il sans ?**

`MaxPooling2D(2,2)` divise chaque dimension par 2 en ne gardant que la valeur maximale dans chaque bloc 2×2. Il remplit deux rôles complémentaires : il **réduit la taille des feature maps** (donc le nombre de paramètres et le coût de calcul), et il introduit une **invariance aux petites translations** — un motif détecté légèrement décalé dans l'image donnera le même résultat après pooling. Sans pooling, les feature maps garderaient leur taille d'origine à chaque couche, la mémoire nécessaire exploserait et le modèle deviendrait beaucoup trop sensible à la position exacte des motifs dans l'image.

**3. Pourquoi `Dropout(0.3)` réduit-il l'overfitting ?**

À chaque batch d'entraînement, Dropout désactive aléatoirement 30 % des neurones — ils ne propagent ni n'apprennent rien pendant ce passage. Cela oblige le réseau à ne pas se reposer sur des neurones spécifiques et à **distribuer la représentation de l'information** sur l'ensemble du réseau. En pratique, c'est comme entraîner plusieurs sous-réseaux légèrement différents à chaque batch, puis combiner leurs prédictions à l'inférence (où Dropout est désactivé). Le réseau ne peut plus mémoriser les exemples d'entraînement via des chemins fixes — il est contraint d'apprendre des représentations plus générales et robustes.

---

## Étape 6 — Courbes d'apprentissage

**1. L'écart train/validation indique-t-il de l'overfitting ?**

Sur le réseau dense (MLP), un léger écart train/validation est généralement visible après quelques époques : l'accuracy de validation plafonne ou commence à décrocher légèrement pendant que l'accuracy d'entraînement continue de progresser — signe d'un début d'overfitting. Le CNN, grâce au Dropout, présente des courbes plus resserrées et une validation plus stable. L'amplitude de cet écart reste modérée sur Fashion-MNIST grâce au volume de données (60 000 exemples), mais elle serait bien plus problématique avec un dataset plus petit.

**2. Le CNN converge-t-il plus vite ou plus lentement que le réseau dense ?**

Le CNN converge **plus rapidement** et atteint une accuracy de validation plus élevée dès les premières époques. En exploitant la structure spatiale des images via ses filtres convolutifs, il extrait des représentations pertinentes dès les premiers passages — là où le MLP doit apprendre des combinaisons de pixels bruts sans aucun a priori spatial, ce qui est intrinsèquement plus lent et moins efficace.

**3. Quel nombre d'époques choisir pour la mise en production ?**

Pour le réseau dense, on choisirait autour de **10–12 époques** — au-delà, le gain de validation stagne et l'overfitting s'installe progressivement. Pour le CNN, **8–10 époques** suffisent : la convergence est plus rapide et la validation se stabilise tôt. Dans les deux cas, on implémenterait un `EarlyStopping(patience=3)` en production pour arrêter automatiquement l'entraînement dès que la val_loss ne s'améliore plus, et un `ModelCheckpoint` pour sauvegarder uniquement les poids correspondant à la meilleure validation.

---

## Étape 7 — Comparaison des 3 approches

**1. Quel modèle atteint l'objectif business (taux d'erreur < 5 %) ?**

| Modèle | Accuracy (typique) | Taux d'erreur | Objectif atteint |
|---|---|---|---|
| Random Forest | ~87 % | ~13 % | ✗ |
| Réseau Dense (MLP) | ~88 % | ~12 % | ✗ |
| CNN | **~91–93 %** | **~7–9 %** | ✗ partiellement |

Le CNN se rapproche le plus de l'objectif mais ne l'atteint pas encore pleinement avec cette architecture de base sur des images 28×28. En production avec des images haute résolution et un CNN plus profond (ResNet, EfficientNet), le taux d'erreur tomberait bien en dessous de 5 %.

**2. Quelles catégories sont le plus souvent confondues ?**

Les confusions les plus fréquentes sont :
- **Pull ↔ Chemise** (classes 2 et 6) : silhouettes similaires, bords peu définis à 28×28
- **T-shirt ↔ Chemise** (classes 0 et 6) : même forme générale de haut
- **Sandale ↔ Sneaker ↔ Bottine** (classes 5, 7, 9) : chaussures aux formes proches à basse résolution

Ces confusions sont cohérentes visuellement — même un humain hésiterait sur certains cas ambigus à cette résolution. Ce n'est pas un dysfonctionnement du modèle, mais une limite inhérente à la qualité des images.

**3. Faut-il uniquement regarder l'accuracy globale en production ?**

Non, absolument pas. L'accuracy globale masque des disparités importantes entre catégories. Un modèle à 91 % d'accuracy globale peut très bien n'avoir que 75 % de recall sur la catégorie "Chemise" — ce qui signifie que 25 % des chemises sont mal catégorisées et partent dans le mauvais rayon. En production, il faut surveiller le **recall** (ne pas rater d'articles d'une catégorie), la **précision** (ne pas mélanger des catégories dans les résultats de recherche) et le **F1-score** par catégorie, en particulier pour les classes qui se ressemblent. Le comité technique doit fixer des seuils minimaux par catégorie, pas seulement une accuracy globale.

---

## Étape 8 — Analyse des erreurs

**1. Les erreurs les plus fréquentes sont-elles compréhensibles visuellement ?**

Oui, entièrement. Les erreurs Pull/Chemise et T-shirt/Chemise sont logiques : ces vêtements partagent une forme rectangulaire avec des manches, et à 28×28 pixels, les détails discriminants (texture du tricot, boutons, col) sont quasi-invisibles. Les confusions entre chaussures (Sandale, Sneaker, Bottine) s'expliquent aussi facilement : la silhouette générale d'une chaussure vue de côté est similaire entre les trois catégories. Ces erreurs sont du même type que celles qu'un humain commettrait sur des thumbnails flous.

**2. Le modèle fait-il des erreurs avec une haute confiance (> 90 %) ? Pourquoi c'est risqué ?**

C'est le scénario le plus problématique en production. Un modèle qui se trompe à 95 % de confiance ne déclenche aucune alerte — l'article est catalogué dans la mauvaise catégorie sans que personne ne remarque quoi que ce soit, puisqu'on lui fait entièrement confiance. Ces erreurs à haute confiance sont généralement des **cas ambigus du dataset** où l'image est effectivement trompeuse, ou des **artefacts visuels** (fond inhabituel, angle de prise de vue atypique) que le modèle n'a pas appris à gérer. En production, il faut mettre en place un **seuil de confiance minimum** (ex. : 85 %) en dessous duquel l'article est renvoyé vers une validation humaine, quelle que soit la prédiction.

**3. Propositions pour les catégories les plus confondues :**

- **Pull ↔ Chemise** : Ajouter un **sous-système de validation humaine** déclenché uniquement pour ces deux catégories quand la confiance est < 90 %. Enrichir les données avec davantage d'exemples de cas limites (pulls à col chemise, chemises en maille). Envisager la **fusion en une catégorie "Haut"** si la distinction n'a pas d'impact fort sur l'expérience utilisateur.
- **Chaussures (Sandale/Sneaker/Bottine)** : Utiliser une **approche hiérarchique** — d'abord détecter la super-catégorie "Chaussure", puis affiner avec un second modèle dédié. Augmenter la résolution d'entrée pour ces catégories spécifiquement.

---

## Étape 9 — Interprétabilité du CNN

**1. Les filtres appris détectent-ils des contours, des textures ou des zones uniformes ?**

Les filtres de la première couche convolutive apprennent principalement des **détecteurs de contours** — horizontaux, verticaux, diagonaux — et des **détecteurs de textures simples**. C'est un comportement universel observé sur toutes les architectures CNN entraînées sur des images : la première couche imite les cellules simples du cortex visuel primaire (V1) chez les mammifères, qui réagissent à des orientations de bords précises. Certains filtres détectent des zones uniformes claires ou sombres, ce qui aide à segmenter les objets du fond.

**2. Sur les feature maps, quels filtres réagissent à la silhouette du sneaker ?**

Les filtres qui produisent les feature maps les plus "allumées" sur les bords de la semelle et de la tige de la chaussure sont typiquement ceux spécialisés dans la **détection de contours obliques ou courbes**. Les filtres qui réagissent fortement sur le fond uniforme correspondent aux **détecteurs de zones homogènes**. En pratique, on observe que 6 à 8 filtres sur 32 activent fortement la silhouette de l'objet, tandis que les autres s'activent sur des textures ou artefacts de fond.

**3. Ces visualisations suffisent-elles pour un comité éthique sur la couleur de fond ?**

Non, les visualisations des filtres et des feature maps ne sont pas suffisantes pour prouver l'absence de biais lié au fond. Elles montrent *ce que* le CNN regarde, mais pas *comment* ça influence les prédictions. Pour répondre sérieusement à cette question, il faudrait compléter par :
- **SHAP ou Grad-CAM** pour visualiser les zones de l'image qui contribuent positivement ou négativement à chaque prédiction
- Des **tests de robustesse** : modifier artificiellement la couleur du fond d'un panel d'images et mesurer si les prédictions changent
- Un **audit sur un sous-ensemble d'images avec différents types de fonds** (blanc, couleur, texturé) pour vérifier que les performances restent homogènes

---

## Étape 10 — Recommandation au comité technique Zalando

### Insight 1 — ML classique vs Deep Learning

**Observation** : Random Forest atteint ~87 % d'accuracy (taux d'erreur ~13 %), le CNN atteint ~91–93 % (taux d'erreur ~7–9 %). L'objectif business de 5 % n'est pas encore atteint par l'architecture de base, mais le CNN s'en approche significativement là où le Random Forest stagne au niveau initial.

**Explication** : Random Forest traite chaque pixel de façon indépendante, sans aucune notion de voisinage ou de structure spatiale. Il est fondamentalement aveugle aux contours, aux formes et aux textures qui caractérisent les vêtements. Le CNN, lui, apprend des hiérarchies de représentations visuelles — d'abord des bords, puis des formes, puis des structures complètes — exactement comme le ferait un opérateur humain qui identifie une robe par sa silhouette, ses bretelles, sa longueur.

**Recommandation** : Le surcoût GPU du CNN est clairement justifié. L'écart de performance est significatif et la différence s'accentuerait encore plus avec des images haute résolution, où Random Forest deviendrait totalement inutilisable. En production, on opterait pour un CNN pré-entraîné (EfficientNetB0 ou ResNet50) sur des images couleur haute résolution, capable d'atteindre des accuracy supérieures à 97 %.

---

### Insight 2 — Dense vs CNN

**Observation** : Le réseau dense (MLP) plafonne à ~88 % d'accuracy, contre ~91–93 % pour le CNN — soit 3 à 5 points d'écart malgré un nombre de paramètres similaire (le MLP en a même davantage). Le CNN atteint ce résultat avec moins d'époques et une meilleure stabilité des courbes de validation.

**Explication** : Le mécanisme clé est le **partage de paramètres** des convolutions. Un filtre 3×3 du CNN apprend à détecter, par exemple, un bord vertical, puis applique ce filtre *partout* dans l'image — que le motif soit en haut à gauche ou en bas à droite. Le MLP, lui, a un neurone différent pour chaque position de l'image, et doit apprendre séparément que "un bord vertical en (5,5)" et "un bord vertical en (15,20)" sont la même chose. C'est une inefficacité fondamentale sur les images.

**Recommandation** : Un réseau dense seul ne devrait jamais être utilisé pour de la classification d'images en production. Il reste pertinent en bout de chaîne (après les couches convolutives) pour la classification finale, mais les couches de feature extraction doivent toujours être convolutives. Pour des données tabulaires pures (sans structure spatiale), le MLP reste en revanche un choix valide.

---

### Insight 3 — Fiabilité en production

**Observation** : Les catégories Chemise, Pull et T-shirt concentrent la majorité des erreurs du CNN, avec des taux de rappel parfois inférieurs à 80 % sur ces classes spécifiques. Des erreurs à haute confiance (> 90 %) existent sur des cas visuellement ambigus — c'est le risque le plus critique pour un déploiement sans supervision.

**Explication** : À 28×28 pixels en niveaux de gris, certaines distinctions entre catégories sont structurellement impossibles à faire avec certitude — les informations discriminantes (boutons, matière, finitions) ne sont tout simplement pas encodées à cette résolution. Le modèle fait les mêmes erreurs qu'un humain ferait sur des vignettes floues. Ce n'est pas un problème d'architecture mais une limite du signal d'entrée.

**Recommandation** : Déployer le CNN avec un **seuil de confiance à 85 %** : en dessous de ce seuil, l'article est automatiquement routé vers une file de validation humaine. Cela permettrait de couvrir automatiquement ~85–90 % des articles uploadés (les cas clairs), tout en garantissant une qualité parfaite sur les 10–15 % restants. À terme, les corrections humaines alimentent une boucle d'apprentissage continu pour améliorer progressivement les performances sur les cas difficiles. Avec des images haute résolution et un modèle pré-entraîné, l'objectif des 5 % d'erreur devient pleinement atteignable.
