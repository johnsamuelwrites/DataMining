# Projet : Système de recommandation d'images

## Présentation

Dans ce projet, vous allez construire un **Système de recommandation d'images** qui suggère des images aux utilisateurs en fonction de leurs préférences. Ce projet applique toutes les compétences que vous avez acquises lors des travaux pratiques : analyse de données, visualisation, clustering, classification et apprentissage automatique.

**Durée** : 3 séances de travaux pratiques
**Taille de l'équipe** : 2-3 étudiants
**Livrables** :
1. Un notebook Jupyter (`Nom1_Nom2_[Nom3].ipynb`)
2. Un rapport de synthèse de 4 pages (PDF)

---

## Objectifs d'apprentissage

En réalisant ce projet, vous allez :
- Automatiser la collecte de données à partir de sources web
- Extraire et traiter les métadonnées des images
- Appliquer des algorithmes de clustering pour analyser les caractéristiques des images
- Construire des profils de préférences utilisateurs
- Implémenter un algorithme de recommandation
- Visualiser efficacement les données
- Écrire des tests complets pour votre système

---

## Architecture du projet

Le système se compose de 7 tâches interconnectées :

```
┌─────────────────────────────────────────────────────────────────┐
│              SYSTÈME DE RECOMMANDATION D'IMAGES                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ 1. Collecte  │───▶│ 2. Étiquetage│───▶│ 3. Analyse   │       │
│  │ de données   │    │ & Annotation │    │ des données  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│        │                    │                   │                │
│        ▼                    ▼                   ▼                │
│  ┌──────────────────────────────────────────────────────┐       │
│  │          Fichiers JSON (Stockage des métadonnées)     │       │
│  └──────────────────────────────────────────────────────┘       │
│        │                    │                   │                │
│        ▼                    ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ 4. Visuali-  │    │ 5. Système   │    │ 6. Tests     │       │
│  │ sation       │    │ de recomman- │    │              │       │
│  └──────────────┘    │ dation       │    └──────────────┘       │
│                      └──────────────┘                            │
│                             │                                    │
│                             ▼                                    │
│                    ┌──────────────┐                              │
│                    │ 7. Rapport   │                              │
│                    │ de synthèse  │                              │
│                    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

![Architecture](../../images/Project-Architecture.png "Architecture")

---

## Tâche 1 : Collecte de données

### Objectif
Collecter au moins **100 images sous licence libre** avec leurs métadonnées.

### Ce que vous devez faire

1. **Créer une structure de dossiers** :
   ```
   projet/
   ├── images/           # Images téléchargées
   ├── data/             # Fichiers JSON de métadonnées
   └── projet.ipynb      # Votre notebook
   ```

2. **Trouver des sources d'images** (choisissez une ou plusieurs) :
   - [Wikimedia Commons](https://commons.wikimedia.org/) - Utilisez des requêtes SPARQL (comme dans le TP 1)
   - [Unsplash API](https://unsplash.com/developers) - API gratuite pour des images de haute qualité
   - [Pexels API](https://www.pexels.com/api/) - Photos gratuites
   - [Flickr API](https://www.flickr.com/services/api/) - Images sous Creative Commons

3. **Télécharger les images programmatiquement** en utilisant les techniques du TP 1, Exercice 6

4. **Extraire et sauvegarder les métadonnées** pour chaque image :
   - Nom du fichier image
   - Dimensions de l'image (largeur, hauteur)
   - Format du fichier (.jpg, .png, etc.)
   - Taille du fichier (en Ko)
   - URL source
   - Informations de licence
   - Données EXIF (si disponibles) : modèle d'appareil photo, date de prise de vue, etc.

### Sortie attendue
- Dossier `images/` avec 100+ images
- `data/images_metadata.json` contenant les métadonnées de toutes les images

### Conseils
- Utilisez `PIL` pour obtenir les dimensions des images
- Utilisez `os.path.getsize()` pour obtenir la taille du fichier
- Utilisez l'extraction EXIF (voir TP 2, Exercice 2)
- Stockez les métadonnées sous forme de liste de dictionnaires au format JSON

---

## Tâche 2 : Étiquetage et annotation

### Objectif
Ajouter des étiquettes descriptives et des caractéristiques calculées à chaque image.

### Ce que vous devez faire

1. **Extraire les informations de couleur** en utilisant le clustering KMeans (TP 2, Exercice 3) :
   - Trouver les 3-5 couleurs prédominantes dans chaque image
   - Stocker les couleurs sous forme de valeurs RGB et/ou de noms de couleurs

2. **Déterminer l'orientation de l'image** :
   - Paysage (largeur > hauteur)
   - Portrait (hauteur > largeur)
   - Carré (largeur ≈ hauteur)

3. **Ajouter des tags de catégorie** (choisissez une approche) :
   - **Manuelle** : Créer une interface simple pour taguer les images
   - **Automatisée** : Utiliser les catégories/tags de la source d'images
   - **Hybride** : Commencer avec les tags sources, permettre un affinage par l'utilisateur

4. **Classifier la taille de l'image** :
   - Vignette : < 500px
   - Moyenne : 500-1500px
   - Grande : > 1500px

### Sortie attendue
- `data/images_labels.json` avec métadonnées enrichies :
```json
{
  "image_001.jpg": {
    "predominant_colors": [[255, 128, 0], [0, 100, 200], [50, 50, 50]],
    "color_names": ["orange", "bleu", "gris"],
    "orientation": "paysage",
    "size_category": "moyenne",
    "tags": ["nature", "coucher de soleil", "plage"]
  }
}
```

### Conseils
- Réutilisez votre code d'extraction de couleurs KMeans du TP 2
- Pensez à utiliser un mapping de noms de couleurs (RGB → nom de couleur)
- Stockez toutes les annotations dans un fichier JSON structuré

---

## Tâche 3 : Analyse des données

### Objectif
Construire des profils de préférences utilisateurs basés sur leurs sélections d'images.

### Ce que vous devez faire

1. **Simuler des utilisateurs** (créez au moins 5 utilisateurs) :
   - Chaque utilisateur "favorite" 10-20 images
   - Les utilisateurs doivent avoir des préférences différentes (un aime la nature, un autre l'architecture, etc.)

2. **Construire des profils utilisateurs** en analysant leurs images favorites :
   ```python
   user_profile = {
       "user_id": "utilisateur_001",
       "favorite_colors": ["bleu", "vert"],       # Couleurs les plus fréquentes
       "favorite_orientation": "paysage",          # Orientation la plus fréquente
       "favorite_size": "moyenne",                 # Taille la plus fréquente
       "favorite_tags": ["nature", "eau"],         # Tags les plus fréquents
       "favorite_images": ["img_01.jpg", ...]      # Liste des images favorites
   }
   ```

3. **Analyser les tendances** entre les utilisateurs :
   - Quelles couleurs sont les plus populaires globalement ?
   - Quels tags apparaissent le plus fréquemment ?
   - Y a-t-il des clusters d'utilisateurs avec des préférences similaires ?

### Sortie attendue
- `data/users.json` avec les profils utilisateurs
- Résultats d'analyse montrant les tendances de préférences utilisateurs

### Conseils
- Utilisez pandas pour l'analyse de données (groupby, value_counts)
- Utilisez Counter de collections pour trouver les éléments les plus fréquents
- Envisagez d'utiliser le clustering pour grouper les utilisateurs similaires

---

## Tâche 4 : Visualisation des données

### Objectif
Créer des visualisations qui révèlent des informations sur votre collection d'images et les préférences des utilisateurs.

### Visualisations requises

1. **Statistiques de la collection d'images** :
   - Diagramme en barres : Nombre d'images par orientation
   - Diagramme en barres : Nombre d'images par catégorie de taille
   - Diagramme circulaire : Distribution des formats d'images

2. **Analyse des couleurs** :
   - Afficher les couleurs prédominantes sous forme de palettes de couleurs
   - Histogramme des fréquences de couleurs sur toutes les images

3. **Préférences utilisateurs** :
   - Diagramme en barres : Couleurs favorites par utilisateur
   - Graphique de comparaison : Préférences des utilisateurs côte à côte

4. **Analyse des tags** :
   - Diagramme en barres : Tags les plus fréquents
   - Nuage de mots (optionnel) : Visualisation de la fréquence des tags

### Sortie attendue
- Au moins 6 visualisations différentes dans votre notebook
- Tous les graphiques doivent avoir des titres, des légendes et des axes étiquetés

### Conseils
- Utilisez matplotlib pour toutes les visualisations (TP 2, Exercice 1)
- Sauvegardez les graphiques importants avec `plt.savefig()`
- Utilisez les subplots pour grouper les visualisations liées

---

## Tâche 5 : Système de recommandation

### Objectif
Implémenter un système qui recommande des images aux utilisateurs en fonction de leurs préférences.

### Choisissez votre approche

Vous devez implémenter **au moins une** de ces approches :

#### Option A : Filtrage basé sur le contenu (utilisant la classification)
Recommander des images similaires à ce que l'utilisateur a déjà aimé.

```python
# Entraîner un classificateur sur les favoris de l'utilisateur
# Caractéristiques : couleur, orientation, taille, tags
# Étiquette : Favori / Non favori
# Prédire quelles images non vues l'utilisateur aimerait
```

**Utiliser** : Decision Tree, Random Forest, ou SVM (TP 3, Exercices 2-3)

#### Option B : Recommandation basée sur le clustering
Grouper les images similaires ensemble et recommander depuis le même cluster.

```python
# Clusterer toutes les images basées sur les caractéristiques
# Trouver à quel cluster appartiennent les favoris de l'utilisateur
# Recommander d'autres images du même cluster
```

**Utiliser** : KMeans (TP 2, Exercices 3-5)

#### Option C : Approche hybride
Combiner les deux méthodes pour de meilleures recommandations.

### Exigences d'implémentation

1. **Entrée** : ID utilisateur
2. **Sortie** : Liste de 5-10 images recommandées (pas déjà favorites)
3. **Explication** : Brève raison pour laquelle chaque image est recommandée

### Sortie attendue
```python
def recommend_images(user_id, n_recommendations=5):
    """
    Recommander des images pour un utilisateur.

    Args:
        user_id: L'utilisateur pour lequel recommander
        n_recommendations: Nombre d'images à recommander

    Returns:
        Liste de tuples (nom_fichier_image, raison)
    """
    # Votre implémentation
    pass
```

### Conseils
- Commencez avec les exemples dans `examples/recommendation.ipynb`
- Utilisez LabelEncoder pour convertir les caractéristiques catégorielles en nombres
- Testez vos recommandations manuellement - ont-elles du sens ?

---

## Tâche 6 : Tests

### Objectif
Vérifier que votre système fonctionne correctement.

### Tests requis

1. **Tests de validation des données** :
   - Toutes les images existent dans le dossier images
   - Toutes les images ont des métadonnées
   - Les valeurs des métadonnées sont valides (pas de dimensions négatives, etc.)

2. **Tests des fonctions** :
   - L'extraction de couleurs retourne des valeurs RGB valides
   - La génération de profil utilisateur fonctionne correctement
   - La fonction de recommandation retourne le nombre de résultats attendu

3. **Tests de qualité des recommandations** :
   - Les images recommandées ne sont pas déjà dans les favoris de l'utilisateur
   - Les recommandations correspondent aux préférences de l'utilisateur (ex. si l'utilisateur aime les images bleues, les recommandations devraient inclure des images bleues)

### Sortie attendue
```python
def test_data_integrity():
    """Tester que toutes les données sont valides"""
    # Vos tests
    assert len(images) >= 100, "Besoin d'au moins 100 images"
    assert all_images_have_metadata(), "Métadonnées manquantes"

def test_recommendation_system():
    """Tester que les recommandations fonctionnent"""
    recommendations = recommend_images("utilisateur_001", 5)
    assert len(recommendations) == 5, "Devrait retourner 5 recommandations"
    # Plus de tests...
```

### Conseils
- Utilisez les instructions `assert` pour des tests simples
- Affichez des messages clairs de succès/échec
- Testez les cas limites (profil utilisateur vide, nouvel utilisateur, etc.)

---

## Tâche 7 : Rapport de synthèse

### Objectif
Écrire un rapport de 4 pages résumant votre projet.

### Sections requises

1. **Introduction** (0,5 page)
   - Objectif du projet
   - Votre approche en bref

2. **Collecte de données** (0,5 page)
   - Sources d'images et licences
   - Nombre d'images collectées
   - Métadonnées stockées

3. **Méthodologie** (1,5 page)
   - Approche d'étiquetage (comment vous avez extrait les caractéristiques)
   - Construction du profil utilisateur
   - Algorithme de recommandation choisi et pourquoi
   - Inclure le diagramme d'architecture

4. **Résultats** (1 page)
   - Visualisations clés (2-3 figures)
   - Précision/qualité des recommandations
   - Découvertes intéressantes

5. **Limitations et travaux futurs** (0,25 page)
   - Qu'est-ce qui n'a pas bien fonctionné ?
   - Comment pourrait-on l'améliorer ?

6. **Conclusion** (0,25 page)
   - Résumé des réalisations
   - Auto-évaluation

### Format
- 4 pages maximum
- Format PDF
- Pas de code dans le rapport (seulement les résultats et explications)
- Inclure les références/bibliographie

---

## Critères d'évaluation

| Tâche | Points | Critères clés |
|-------|--------|---------------|
| Collecte de données | 15% | Automatisation, 100+ images, métadonnées complètes |
| Étiquetage & Annotation | 15% | Extraction de couleurs, catégorisation appropriée |
| Analyse des données | 15% | Profils utilisateurs, analyse des préférences |
| Visualisation des données | 15% | 6+ visualisations, formatage correct |
| Système de recommandation | 20% | Algorithme fonctionnel, recommandations raisonnables |
| Tests | 10% | Tests complets, tous passent |
| Rapport de synthèse | 10% | Clair, complet, bien structuré |

---

## Soumission

### Fichiers à soumettre
```
Nom1_Nom2_[Nom3].zip
├── Nom1_Nom2_[Nom3].ipynb    # Votre notebook
├── data/
│   ├── images_metadata.json
│   ├── images_labels.json
│   └── users.json
└── rapport_synthese.pdf
```

### Notes importantes
- **NE SOUMETTEZ PAS** le dossier images (trop volumineux)
- Assurez-vous que votre notebook s'exécute sans erreurs
- Incluez des commentaires expliquant votre code
- Renommez les fichiers avec les noms des membres de votre équipe

---

## Pour commencer

1. Commencez avec le notebook template : `fr/Projet/projet.ipynb`
2. Consultez les exemples dans `examples/recommendation.ipynb`
3. Réutilisez le code de vos travaux pratiques
4. Travaillez progressivement - complétez chaque tâche avant de passer à la suivante

**Bon courage !**
